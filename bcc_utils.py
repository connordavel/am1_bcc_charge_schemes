# utils to help with visualization and manipulation of molecules. 
# also contains modifications to the openff toolkit to assist with the project, but do not change the toolkit itself 
import sys, getopt
if __name__ == "__main__":
    args = sys.argv
    if len(args) > 1:
        try:
            opts, args = getopt.getopt(args[1:], "ht:n:", ['task=', 'total_tasks='])
        except getopt.GetoptError:
            print('bcc_utils.py --task <task num> --total_tasks <total number of tasks>')
            sys.exit(2)
        for opt, arg in opts:
            if opt == '-h':
                print('bcc_utils.py --task <task num> --total_tasks <total number of tasks>')
                sys.exit()
            elif opt in ['--task', '-t']:
                task_number = int(arg)
            elif opt in ['--total_tasks', '-n']:
                total_tasks = int(arg)
    else:
        task_number = 0
        total_tasks = 1

import os
import re
import numpy
from pathlib import Path
import pandas as pd
import subprocess
from typing_extensions import get_args
import py3Dmol
from openff.toolkit.topology.molecule import FrozenMolecule, Molecule
from openff.toolkit.utils.toolkits import RDKitToolkitWrapper, OpenEyeToolkitWrapper
from qcelemental.molutil import guess_connectivity
from distutils.spawn import find_executable
import tempfile
from openff.toolkit.utils.utils import temporary_cd   # this is really cool btw
from simtk.openmm.app import Element
from rdkit import Chem
from copy import deepcopy
import shutil
from tempfile import TemporaryDirectory
from openeye import oechem
from time import time
import datetime
from joblib import Parallel, delayed
from collections import defaultdict
from simtk.openmm.app import PDBFile
from simtk import openmm, unit
from openff.toolkit.topology import Topology
from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.toolkit.utils import UndefinedStereochemistryError
from run import *
from geometric.engine import Engine
from geometric.internal import DelocalizedInternalCoordinates, CartesianCoordinates
from geometric.molecule import Molecule as GeoMolecule
from geometric.nifty import ang2bohr, bohr2ang
from geometric.optimize import Optimize, OptParams, ParseConstraints
from geometric.errors import GeomOptNotConvergedError

# call antechamber and return the final molecule after am1-bcc simulation
# see AmberToolsToolkitWrapper.assign_partial_charges() for the original version
# of this code. 

class MolDatabase:
    """
    Iterator class that stores the location of multiple molecule and protein folders and files.
    The locations of these files are hard coded in the __init__(self) function and must be changed
    manually to accomodate each specific machine. May add functionality to change this in the future
    """
    def __init__(self, max_mol_size=100, only_smiles=False, select_mols=[], task=0, total_tasks=1):
        # the location of the files and what they are (ie. a folder or a file)
        # if the entry is a file, we open the file and yield the individual molecules in
        # that file. If the entry is a folder, we yield all molecule files in that folder
        # the file type is also included as an added check and to help with folder parsing 
        # the "proteins" file type is a special case we use for the amber-ff-porting protein database
        # which will seach for specific conbinations of positive and negative type proteins
        
        # currently, only sdf files are supported, but more formats can be added as needed 
        if only_smiles:
            self.files = [("smiles", "smiles", None)]
        else:
            self.files = [
                        ("file", "sdf", Path( Path.cwd() / "MiniDrugBank.sdf")),
                        #   ("file", "sdf", Path( Path.cwd() / "burn-in.sdf")),
                        ("protein", "mol2", Path("/home/coda3831/openff-workspace/amber-ff-porting/parameter_deduplication/amber-ff-porting")),
                        ("smiles", "smiles", Path())
                        ]
        self.max_mol_size = max_mol_size
        self.select_mols = select_mols
        self.task = task
        self.total_tasks = total_tasks

    def failed(self):
        off_mol = Molecule()
        off_mol.name = "failed"
        return off_mol, -1

    def letter_index(self, num):
        indexes = []
        while num >= 26:
            indexes.append(num % 26)
            num = int(num / 26) - 1
        indexes.append(num)
        indexes.reverse()
        string = ""
        for i in indexes:
            string += chr(ord('a') + i)
        return string
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # molecule suppliers (basically just Molecule.from_file() but with error handling
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    def sdf_mol_file_supplier(self, file_path):
        """
        Takes as input the file path to an sdf file. Yields each molecule in the sdf file using the 
        openeye toolkit. 
        returns:
            off_molecule, 1 for a successful molecule
            off_molecule, -1 for a failed molecule. 
        """
        if file_path.suffix != ".sdf":
            print("incorrect file type in sdf_mol_file_supplier")
            yield self.failed()
        if not isinstance(file_path, Path):
            file_path = Path(file_path)
        try:
            for off_mol in Molecule.from_file(str(file_path), file_format="SDF", allow_undefined_stereo=True):
                try:
                    Molecule.from_smiles(off_mol.to_smiles())
                except UndefinedStereochemistryError:
                    off_mol = [off_mol, *off_mol.enumerate_stereoisomers()][-1]
                if off_mol.n_atoms == 0:
                    yield self.failed()
                elif off_mol.n_atoms <= self.max_mol_size:
                    yield off_mol, 1
        except Exception as e:
            print(f"major exception for {file_path.name}")
            print(e)
            yield self.failed()

    def sdf_mol_folder_supplier(self, folder_path):
        # implement this when it is needed
        # also, possibly change name to "mol_folder_supplier" and automatically
        # return molecules whether in mol2, sdf, pdb, etc? 
        return None

    def amber_protein_mol_supplier(self, file_path):
        def fix_carboxylate_bond_orders(offmol):
            # function provided by Jeffrey Wagner
            """Fix problem where leap-produced mol2 files have carboxylates defined with all single bonds"""
            # First, find carbanions
            for atom1 in offmol.atoms:
                if atom1.atomic_number == 6 and atom1.formal_charge.value_in_unit(unit.elementary_charge) == -1:
                    # Then, see if they're bound to TWO oxyanions
                    oxyanion_seen = False
                    for bond in atom1.bonds:
                        atom2 = [atom for atom in bond.atoms if not atom == atom1][0]
                        if atom2.element.atomic_number == 8 and atom2.formal_charge.value_in_unit(unit.elementary_charge) == -1:
                            # If we find a bond to a SECOND oxyanion, then zero both 
                            # the carbon and the second oxygen's formal charges, and 
                            # set the bond order to 2
                            if oxyanion_seen:
                                atom1._formal_charge = 0 * unit.elementary_charge
                                atom2._formal_charge = 0 * unit.elementary_charge
                                bond._bond_order = 2
                            oxyanion_seen = True
       
        if not isinstance(file_path, Path):
            file_path = Path(file_path)
        if not file_path.exists():
            print("directory does not exists")
            yield self.failed()
        # look for the first combination of the following protein groups
        file_names = []
        pos_proteins = ['LYS', 'HIP', 'ARG']
        neg_proteins = ['GLU', 'ASP']
        for pos in pos_proteins:
            for neg in neg_proteins:
                file_names.append(f"{pos}_{neg}.mol2")
                file_names.append(f"{neg}_{pos}.mol2")
        for file in file_path.glob("**/*.mol2"):
            if file.name in file_names:
                try:
                    off_mol = Molecule.from_file(str(file), file_format="MOL2", allow_undefined_stereo=True)
                    try:
                        Molecule.from_smiles(off_mol.to_smiles())
                    except UndefinedStereochemistryError:
                        off_mol = [off_mol, *off_mol.enumerate_stereoisomers()][-1]
                    if off_mol.n_atoms == 0:
                        yield self.failed()
                    elif off_mol.n_atoms <= self.max_mol_size:
                        off_mol.name = str(file.parents[1]).split("/")[-1]
                        off_mol.name += "_"
                        off_mol.name += file.stem
                        fix_carboxylate_bond_orders(off_mol)
                        yield off_mol, 1
                except Exception as i:
                    print(f"failed to read protein {file.name}")
                    yield self.failed()
            else:
                continue

    def smiles_mol_supplier(self):
        smiles = ['[H]c1c(c(c(c(c1[H])C(=O)N2C(C(C(C2([H])[H])([H])N([H])C(=O)N([H])[H])([H])[H])([H])[H])S(=O)C(F)(F)F)[H])[H]',
                  '[H]C1([C@]2([C@@]1(C(N(C2([H])[H])C(=O)OC(C([H])([H])[H])(C([H])([H])[H])C([H])([H])[H])([H])[H])C(F)(F)F)C([H])([H])N(C([H])([H])[H])C([H])([H])C([H])([H])O[H])[H]',
                  '[H]c1c(c(c(nc1[H])C(=O)N(C([H])([H])[H])C([H])(C([H])([H])C([H])([H])[H])C([H])([H])SC([H])([H])[H])SC([H])([H])C([H])([H])[H])[H]',
                  '[H]c1c(c(c2c(c1[H])C3(C(C(O2)(N(C(=O)N3[H])C([H])([H])[H])C([H])(C([H])([H])[H])C([H])([H])[H])([H])[H])[H])[H])[H]',
                  '[H]c1c(c(c(c(c1[H])[H])C2=NSc3c2nc(nc3N4C(C(SC(C4([H])C([H])([H])O[H])([H])[H])([H])[H])([H])[H])[H])[H])[H]',
                  '[H]C1=C(SC(=N1)C([H])([H])N(C(=O)C2=C(N=NS2)C([H])([H])C([H])([H])[H])C3(C(C34C(C(N(C(C4([H])[H])([H])[H])[H])([H])[H])([H])[H])([H])[H])[H])[H]',
                  '[H]c1c(c(c(c(c1[H])Cl)[H])C2=C(SC(=N2)C([H])([H])N3C(C(C(C3([H])C(=O)[O-])([H])[H])([H])[H])([H])[H])[H])[H].[K+]',
                  '[H]C1=C(SC(=N1)C([H])([H])C([H])([H])[H])C([H])([H])N2C(C(C(C(C2([H])[H])([H])C3=NC(=C(S3)[H])C(=O)O[H])([H])[H])([H])[H])([H])[H]',
                  '[H]c1c(c(c(c(c1[H])C2(C(C(C(N2C([H])([H])C3=NN4C(=O)C(=C(N=C4S3)C([H])([H])[H])[H])([H])[H])([H])[H])([H])[H])[H])Br)[H])[H]',
                  '[c:1]1([H:17])[c:4]([H:5])[c:2]([H:18])[c:7]([C:13]2([H:28])[C:9]([H:20])([H:21])[C:11]([H:24])([H:25])[N+:14]([H:29])([H:30])[C:12]([H:26])([H:27])[C:10]2([H:22])[H:23])[c:3]([H:19])[c:6]1[C:8]([O-:15])=[O:16]'
                 ]
        for smiles_str in smiles:
            try:
                #undefined stereochemistry errors happening here
                off_mol = Molecule.from_smiles(smiles_str, allow_undefined_stereo=True, toolkit_registry=OpenEyeToolkitWrapper())
                # see Simon's find-failures.py script for this and other smiles behavior
                try:
                    Molecule.from_smiles(off_mol.to_smiles())
                except UndefinedStereochemistryError:
                    off_mol = [off_mol, *off_mol.enumerate_stereoisomers()][-1]
                if off_mol.n_atoms == 0:
                    yield self.failed()
                elif off_mol.n_atoms <= self.max_mol_size:
                    off_mol.name = "smiles"
                    yield off_mol, 1
            except Exception as e:
                print(e)
                yield self.failed()

    def __iter__(self):
        # go through all file/folders, return molecules through function generator for each file
        c = 0
        task_index = self.total_tasks
        for file in self.files:
            type, extension, file_path = file
            print(f"parsing {file_path}")
            if type == "file":
                if extension == "sdf":
                    for off_mol, status in self.sdf_mol_file_supplier(file_path):
                        if status == 1:
                            off_mol.name = off_mol.name + f"-{self.letter_index(c)}"
                            c += 1
                            for protomer in [off_mol, *off_mol.enumerate_protomers()]:
                                if (len(self.select_mols) == 0) or (protomer.name in self.select_mols):
                                    if ((task_index-self.task)%self.total_tasks == 0):
                                        yield protomer, file_path.name
                                    task_index += 1
                        else:
                            print(f"failed molecule in {file_path.name}")
                else:
                    print(f"file type not supported for {file_path.name}")
                    continue
            elif type == "folder":   # not implemented 
                continue
            elif type == "protein":
                # special case for the protein database provided by Jeff in amber-ff-porting 
                for off_mol, status in self.amber_protein_mol_supplier(file_path):
                    if status == 1:
                        off_mol.name = off_mol.name + f"-{self.letter_index(c)}"
                        c += 1
                        protomers = [off_mol, *off_mol.enumerate_protomers()]
                        # only use first 4 protomers for proteins, for time savings
                        for protomer in protomers[:4]:
                            if (len(self.select_mols) == 0) or (protomer.name in self.select_mols):
                                if ((task_index-self.task)%self.total_tasks == 0):
                                    yield protomer, file_path.name
                                task_index += 1
                    else:
                        print(f"failed molecule in {file_path.name}")
            elif type =="smiles":
                for off_mol, status in self.smiles_mol_supplier():
                    if status == 1:
                        off_mol.name = off_mol.name + f"-{self.letter_index(c)}"
                        c += 1
                        for protomer in [off_mol, *off_mol.enumerate_protomers()]:
                            if (len(self.select_mols) == 0) or (protomer.name in self.select_mols):
                                if ((task_index-self.task)%self.total_tasks == 0):
                                    yield protomer, file_path.name
                                task_index += 1
                    else:
                        print(f"failed molecule in smiles")

            else:
                print(f"no known file formats to parse {file_path.name}")
                continue 

def run_am1_bcc(mol, output_dir, arguments=[]):
    """
    mol: off molecule to be run through antechamber am1-bcc simulation
    output_dir: name of directory to place results of antechamber simultion
    return_offmol: True to return an offmol representation of the antechamber simulation result
    returns: A list of success/failure statuses for each conformer run
            True if am1_bcc simulation ran without failure, False if molecule has changed connectivity
    """
    def simulate_conformer(conformer, conf_idx):
        # create the output directory if it doesn't already exist
        Path(Path.cwd() / output_dir / f"conf{conf_idx}").mkdir(parents=True, exist_ok = True)
        with temporary_cd(output_dir + f"/conf{conf_idx}"):
            net_charge = mol_copy.total_charge / unit.elementary_charge
            mol_copy._conformers = [conformer]
            # Write out molecule in SDF format
            OpenEyeToolkitWrapper().to_file(
                mol_copy, file_path=f"input_{mol.name}.sdf", file_format="sdf"
            )
            inputs = [
                    "antechamber",
                    "-i",
                    f"input_{mol.name}.sdf",
                    "-fi",
                    "sdf",
                    "-o",
                    f"output_{mol.name}.mol2",
                    "-fo",
                    "mol2",
                    "-pf",
                    "yes",
                    "-dr",
                    "n",
                    "-c",
                    short_charge_method,
                    "-nc",
                    str(net_charge),
                ]
            for arg in arguments:
                inputs.append(arg)

            try:
                subprocess.check_output(inputs)

                # output charges for later comparison
                subprocess.check_output(
                    [
                        "antechamber",
                        "-dr",
                        "n",
                        "-i",
                        f"output_{mol.name}.mol2",
                        "-fi",
                        "mol2",
                        "-o",
                        f"output_{mol.name}.mol2",
                        "-fo",
                        "mol2",
                        "-c",
                        "wc",
                        "-cf",
                        f"charges_{mol.name}.txt",
                        "-pf",
                        "yes",
                    ]
                )
            except Exception:
                return {"openeye": "run_failed", "rdkit": "run_failed", "ambertools": "run_failed"}
            # read output into new molecule and conpare to the input molecule with isomorphism
            # we use 3 different methods and output the results of all of them in the form of a dictionary
            # these methods use openeye, rdkit, and ambertools to interpret the pdb file (and .out file in
            # the case of ambertools)
            isomorphism_results = {}

            # rdkit method
            rdmol = Chem.MolFromPDBFile('sqm.pdb', removeHs=False)
            new_mol = Molecule.from_rdkit(rdmol, allow_undefined_stereo=True, hydrogens_are_explicit=True)
            are_isomorphic, _ = Molecule.are_isomorphic(new_mol, mol_copy,
                                                    aromatic_matching=False,
                                                    formal_charge_matching=False,
                                                    bond_order_matching=False,
                                                    atom_stereochemistry_matching=False,
                                                    bond_stereochemistry_matching=False,
                                                    strip_pyrimidal_n_atom_stereo=True)
            isomorphism_results['rdkit'] = are_isomorphic
            # qcelemental method
            expected_connectivity = {
                tuple(sorted([bond.atom1_index, bond.atom2_index]))
                for bond in mol_copy.bonds
            }
            rdmol = Chem.MolFromPDBFile('sqm.pdb', removeHs=False)
            final_mol = Molecule.from_rdkit(rdmol, allow_undefined_stereo=True, hydrogens_are_explicit=True)
            final_conformer = final_mol._conformers[0]
            new_mol = deepcopy(mol_copy)
            new_mol._conformers = [final_conformer]
            qc_molecule = new_mol.to_qcschema()

            actual_connectivity = {
                tuple(sorted(connection))
                for connection in guess_connectivity(
                    qc_molecule.symbols,
                    qc_molecule.geometry,
                    threshold=1.2
                )
            }
            are_isomorphic = (expected_connectivity == actual_connectivity)
            isomorphism_results['qcelemental'] = are_isomorphic
            # for ease later on, rename intermediate files according to molecule name
            cwd = Path.cwd()
            Path(cwd / f"sqm.in").rename(cwd / f"sqm_{mol.name}.in")
            Path(cwd / f"sqm.out").rename(cwd / f"sqm_{mol.name}.out")
            Path(cwd / f"sqm.pdb").rename(cwd / f"sqm_{mol.name}.pdb")

        return isomorphism_results

    if not isinstance(mol, FrozenMolecule):
        raise TypeError(
            "\"mol\" argument must of class openff.toolkit.topology.Molecule \
            OpenEye and RDkit molecules are not accepted"
        )
    short_charge_method = "bcc"
    ANTECHAMBER_PATH = find_executable("antechamber")
    if ANTECHAMBER_PATH is None:
        raise FileNotFoundError(
            "Antechamber not found, cannot run "
            "AmberToolsToolkitWrapper.assign_fractional_bond_orders()"
        )
    # create copy, In future, can add conformers functionality like in the main toolkit
    mol_copy = Molecule(mol)
    
    # run simulation for each conformer
    if mol_copy.n_conformers == 0:
        conformer_method(mol_copy)
    if mol_copy.n_conformers == 0:
        raise Exception(
            "cannot generate conformers of molecule"
        )
    conformers = deepcopy(mol_copy.conformers)
    # for conformer in conformers:
        # status[f"conf{conf_idx}"] = simulate_conformer(conformer, conf_idx)
        # conf_idx += 1
    
    status = pd.DataFrame(columns = ['rdkit', 'qcelemental'])
    for conformer, conf_idx in zip(conformers, range(0, len(conformers))):
        results = simulate_conformer(conformer, conf_idx)
        status = status.append(results, ignore_index=True)
    # can't get this to work for more than a single core
    # results = Parallel(n_jobs=1)(delayed(simulate_conformer)(conformer, conf_idx) for conformer, conf_idx in zip(conformers, range(0, len(conformers))))
    
    status.reset_index(drop=True, inplace=True)
    return status

def mol_file_viewer(input_file, highlight_idx=[]):
    # only for use in jupyter notebook

    file_format = input_file.split(".")
    file_format = file_format[-1]
    accepted_formats = ['mol2', 'pdb', 'sdf']
    formats = {'mol2': 'MOL2', 'pdb': 'PDB', 'sdf': 'SDF'}
    if file_format not in accepted_formats:
        raise Exception(
            f"file format \"{file_format}\" not accepted"
            )
    # create a molecule
    if file_format == 'pdb':
        mol = Molecule.from_file(input_file, file_format=formats[file_format], allow_undefined_stereo=True)
        rdmol = Chem.MolFromPDBFile(input_file, removeHs=False)
    else:
        mol = Molecule.from_file(input_file, file_format=formats[file_format], allow_undefined_stereo=True)
        rdmol = mol.to_rdkit()
    # create rdkit molecule and extract the mol block
    mb = Chem.MolToMolBlock(rdmol)

    # use mol block as input to the py3Dmol grapher 

    viewer = py3Dmol.view(width=400,height=300)
    viewer.addModel(mb, 'sdf')
    viewer.setStyle({'stick':{}})
    idx = 0
    for atom in mol.atoms:
        if idx in highlight_idx:
            viewer.setStyle({'serial': idx}, {'stick':{'color': 'yellow'}})
        idx += 1

    viewer.zoomTo()
    return viewer

def generage_simulation_frames_sqm(molecule, output_dir, step_mult=1, select_confs=range(0,1000)):
    '''
    in output dir, 3 subdirectories are created to store the frames from the am1
    simulation. These directories contain the .pdb, .out, and charge .txt files 
    that can be used to track the simulation over the course of its run.
    Credit to Simon Boothroyd for the following code on how to create and 
    run sqm files. 
    '''
    def write_in(symbols, x, input_path, step, net_charge):
        # format the coordinates
        if isinstance(x, unit.quantity.Quantity):
            x = x._value
        # Create the input file
        lines = [
            "Single point evaluation",
            " &qmmm",
            "    qm_theory='AM1',",
            "    grms_tol=0.0005,",
            "    scfconv=1.d-10,",
            "    ndiis_attempts=700,",
            f"    ntpr={step}",
            f"    qmcharge={int(net_charge)},",
            f"    maxcyc={step},",
            " /"
        ]
        element_counter = defaultdict(int)

        for symbol, coordinate in zip(symbols, x):
            atomic_number = Element.getBySymbol(symbol).atomic_number
            element_counter[atomic_number] += 1
            atom_type = f"{symbol}{element_counter[atomic_number]}"
    
            lines.append(
                            f"{atomic_number:4d}"
                            f" "
                            f"{atom_type:>4s}"
                            f" "
                            f"{coordinate[0]: .10f}"
                            f" "
                            f"{coordinate[1]: .10f}"
                            f" "
                            f"{coordinate[2]: .10f}"
                            f" "
            )

        with open(input_path, "w") as file:
            file.write("\n".join(lines))

    def read_out(output_path, first_iter=False):
        # reads the molecule coordinates from the .out file
        with open(output_path) as file:
            file_read = file.read()

        output = file_read.split(" Final Structure")

        output_lines = output[1].split("\n")
        coords = []
        for line in filter(lambda line: line.startswith("  QMMM:  "), output_lines):
            nums = line.split()
            nums = nums[-3:]
            floats = [float(i) for i in nums]
            coords.append(floats)
        if not first_iter:
            output_lines = file_read.split("\n")
            energy_line = None
            for line in filter(lambda line: line.startswith("xmin "), output_lines):
                energy_line = line
            
            if energy_line == None:
                energy = 999
                rms = 999
            else:
                energy_line = energy_line.split()
                energy = float(energy_line[2])
                rms = float(energy_line[4])
        else:
            energy = None
            rms = None

        output = file_read.split("  Mulliken Charge")
        output_lines = output[1].split("\n")
        charges = []
        num_atoms = len(coords)
        for line in output_lines[1:num_atoms + 1]:   # +1 for base 0 indexing
            nums = line.split()
            charges.append(float(nums[2]))

        return coords, energy, rms, charges

    def file_op():
        pre_bcc_path = "tmp/pre_bcc.ac"
        pre_bcc_charged_path = "tmp/pre_bcc_charged.ac"
        post_bcc_path = "tmp/post_bcc.ac"
        final_bcc_path = "tmp/final_bcc.ac"
        # create ac file and insert charges from out file (can antechamber do this automatically?)
        subprocess.check_output(["antechamber", "-i", sqm_output_path, "-fi", "sqmout", 
                                "-o", pre_bcc_path, "-fo", "ac", "-dr", "n", "-pf", "y"])
        new_lines = []
        with open(pre_bcc_path) as file:
            lines = file.read().split("\n")
        charge_idx = 0
        for line in lines:
            if line.startswith("ATOM"):
                info = re.split('(\s+)', line)
                if charges[charge_idx] < 0:
                    info[-4] = " "
                info[-3] = f"{charges[charge_idx]:.6f}"
                charge_idx += 1
                line = "".join(info)
            new_lines.append(line)
        with open(pre_bcc_charged_path, "w") as file:
            file.write("\n".join(new_lines))
        # run am1bcc to get new charges
        subprocess.check_output(["am1bcc", "-i", pre_bcc_charged_path, "-o", post_bcc_path, 
                                "-f", "ac", "-p", "/home/coda3831/anaconda3/envs/openff-dev2/dat/antechamber/BCCPARM.DAT",
                                "-j", "1"])
        subprocess.check_output(["atomtype", "-i", post_bcc_path, "-o", final_bcc_path, 
                                "-f", "ac", "-p", "gaff"])
        
        pdb_path = f"tmp/{molecule.name}_{i:05d}.pdb"
        sdf_path = f"tmp/{molecule.name}_{i:05d}.sdf"
        charges_txt_path = f"tmp/charges_{molecule.name}_{i:05d}.txt"
        # write to pdb and write charges to txt
        subprocess.check_output(["antechamber", "-i", final_bcc_path, "-fi", "ac", 
                                "-o", pdb_path, "-fo", "pdb", "-dr", "n",
                                "-c", "wc", "-cf", charges_txt_path, "-pf", "y"])
        # write to sdf
        subprocess.check_output(["antechamber", "-i", final_bcc_path, "-fi", "ac", 
                                "-o", sdf_path, "-fo", "sdf", "-dr", "n", "-pf", "y"])
        # copy all files to their respective directories

        shutil.copy(charges_txt_path, dest_dir / "charge_frames")
        shutil.copy(pdb_path, dest_dir / "pdb_frames")
        shutil.copy(sdf_path, dest_dir / "sdf_frames")
        shutil.copy(sqm_output_path, dest_dir / "out_frames")

        # special append for sdf frames 
        with open(sdf_path, "r") as file:
            sdf_info = file.read()
        sdf_info = sdf_info  + "\n\n$$$$\n"

        open(dest_dir / f"{molecule.name}_{conf_idx}_animation.sdf", "a").write(sdf_info)

        # remove copied files 
        os.remove(charges_txt_path)
        os.remove(pdb_path)
        os.remove(sdf_path)
        os.remove(sqm_output_path)

    with temporary_cd(output_dir):
        mol_copy = Molecule(molecule)
        
        # run simulation for each conformer
        if mol_copy.n_conformers == 0:
            conformer_method(mol_copy)
        if mol_copy.n_conformers == 0:
            raise Exception(
                "cannot generate conformers of molecule"
            )
        conformers = deepcopy(mol_copy.conformers)

        # make the necessary folders
        folders = ['pdb_frames', 'charge_frames', 'sdf_frames', 'out_frames']
        for conf_idx in range(0, len(conformers)):
            # create the output directory if it doesn't already exist
            for folder in folders:
                Path(Path.cwd() / f"conf{conf_idx}" / folder).mkdir(parents=True, exist_ok = True)

        # define symbols that will remain the same
        if isinstance(molecule, Molecule):
            symbols = [atom.element.symbol for atom in molecule.atoms]
        else:
            symbols = molecule.Data["elem"]

        for conformer, conf_idx in zip(conformers, range(0, len(conformers))):
            # so we can select certain conformers
            if conf_idx not in select_confs:
                continue
            dest_dir = Path.cwd() / f"conf{conf_idx}"
            open(dest_dir / f"{molecule.name}_{conf_idx}_animation.sdf", "w").write("")
            net_charge = mol_copy.total_charge / unit.elementary_charge

            tmp_dir = Path(Path.cwd() / "tmp")
            tmp_dir.mkdir(exist_ok = True)
            i = 0
            sqm_input_path = "tmp/sqm.in"
            sqm_output_path = f"tmp/sqm_{i:05d}.out"
            # special case to see the the maxcyc=0 first option
            
            write_in(symbols, conformer, sqm_input_path, 0, net_charge)
            subprocess.check_output(["sqm", "-i", sqm_input_path, "-o", sqm_output_path, "-O"])
            x, nrg, gradient_rms, charges = read_out(sqm_output_path, first_iter=True)
            file_op()

            gradient_tol = 0.0006
            func_max = 0.009
            one_step_grad = float(1)  # gradient above which step is 1
            max_step = float(20)   # step is never larger than this
            m = 1/func_max/max_step / step_mult
            b = 1 - (1/m)

            maxcyc = 1000
            rolling_grads = numpy.array([gradient_rms, 0], dtype=numpy.double)
            i = 1
            while True:
                # quick random function I came up with to speed up the step process
                step = one_step_grad / numpy.std(rolling_grads) * (1/m) + b
                print(step)
                if numpy.isnan(step):
                    step = 1
                step = int(step)
                if step < 1:
                    step = 1
                elif step > 20:
                    step = 20
                print(step)

                sqm_output_path = f"tmp/sqm_{i:05d}.out"
                write_in(symbols, x, sqm_input_path, step, net_charge)
                subprocess.check_output(["sqm", "-i", sqm_input_path, "-o", sqm_output_path, "-O"])
                x, nrg, gradient_rms, charges = read_out(sqm_output_path)

                file_op()

                # if the energy converges immediately with no output, the minimization
                # is finished
                if nrg == 999 and gradient_rms == 999:
                    break

                rolling_grads = numpy.append(rolling_grads, gradient_rms)
                while len(rolling_grads) > 5:
                    rolling_grads = numpy.delete(rolling_grads, 0)
                i += step
                if i % 3 == 0:
                    print(f"gradient: {gradient_rms}\tstdev: {numpy.std(rolling_grads)}")
                if numpy.std(rolling_grads) <= gradient_tol or i >= maxcyc:
                    break

def conformer_method(mol, method="rdkit"):
    if method=="rdkit":
        try:
            mol.generate_conformers(n_conformers=5, toolkit_registry = RDKitToolkitWrapper())
        except Exception:
            print(f"\tRDKit also could not generate conformers for {mol.name}. Continuing with single conformer")
        if len(mol.conformers) == 1:
            try:
                mol.generate_conformers(n_conformers=5, toolkit_registry = OpenEyeToolkitWrapper())
            except Exception as e:
                print(e)
    elif method=="openeye":
        # method used by Simon
        mol.generate_conformers(n_conformers=4, rms_cutoff=1.0 * unit.angstrom)

def my_assign_partial_charges(
        self,
        molecule,
        partial_charge_method=None,
        use_conformers=None,
        strict_n_conformers=False,
        _cls=None,
    ):
        """
        Compute partial charges with OpenEye quacpac, and assign
        the new values to the partial_charges attribute.

        .. warning :: This API is experimental and subject to change.

        .. todo ::

           * Should the default be ELF?
           * Can we expose more charge models?


        Parameters
        ----------
        molecule : openff.toolkit.topology.Molecule
            Molecule for which partial charges are to be computed
        partial_charge_method : str, optional, default=None
            The charge model to use. One of ['amberff94', 'mmff', 'mmff94', `am1-mulliken`, 'am1bcc',
            'am1bccnosymspt', 'am1bccelf10']
            If None, 'am1-mulliken' will be used.
        use_conformers : iterable of simtk.unit.Quantity-wrapped numpy arrays, each with shape (n_atoms, 3) and dimension of distance. Optional, default = None
            Coordinates to use for partial charge calculation. If None, an appropriate number of conformers will be generated.
        strict_n_conformers : bool, default=False
            Whether to raise an exception if an invalid number of conformers is provided for the given charge method.
            If this is False and an invalid number of conformers is found, a warning will be raised.
        _cls : class
            Molecule constructor

        Raises
        ------
        ChargeMethodUnavailableError if the requested charge method can not be handled by this toolkit

        ChargeCalculationError if the charge method is supported by this toolkit, but fails
        """

        import numpy as np
        from openeye import oechem, oequacpac

        from openff.toolkit.topology import Molecule

        SUPPORTED_CHARGE_METHODS = {
            "am1bcc": {
                "oe_charge_method": oequacpac.OEAM1BCCCharges,
                "min_confs": 1,
                "max_confs": 1,
                "rec_confs": 1,
            },
            "am1-mulliken": {
                "oe_charge_method": oequacpac.OEAM1Charges,
                "min_confs": 1,
                "max_confs": 1,
                "rec_confs": 1,
            },
            "gasteiger": {
                "oe_charge_method": oequacpac.OEGasteigerCharges,
                "min_confs": 0,
                "max_confs": 0,
                "rec_confs": 0,
            },
            "mmff94": {
                "oe_charge_method": oequacpac.OEMMFF94Charges,
                "min_confs": 0,
                "max_confs": 0,
                "rec_confs": 0,
            },
            "am1bccnosymspt": {
                "oe_charge_method": oequacpac.OEAM1BCCCharges,
                "min_confs": 1,
                "max_confs": 1,
                "rec_confs": 1,
            },
            "am1bccnoopt": {
                "oe_charge_method": oequacpac.OEAM1BCCCharges,
                "min_confs": 1,
                "max_confs": 1,
                "rec_confs": 1,
            },
            "am1elf10": {
                "oe_charge_method": oequacpac.OEELFCharges(
                    oequacpac.OEAM1Charges(optimize=True, symmetrize=True), 10
                ),
                "min_confs": 1,
                "max_confs": None,
                "rec_confs": 500,
            },
            "am1bccelf10": {
                "oe_charge_method": oequacpac.OEAM1BCCELF10Charges,
                "min_confs": 1,
                "max_confs": None,
                "rec_confs": 500,
            },
        }

        if partial_charge_method is None:
            partial_charge_method = "am1-mulliken"

        partial_charge_method = partial_charge_method.lower()

        if partial_charge_method not in SUPPORTED_CHARGE_METHODS:
            print(
                f"partial_charge_method '{partial_charge_method}' is not available from OpenEyeToolkitWrapper. "
                f"Available charge methods are {list(SUPPORTED_CHARGE_METHODS.keys())} "
            )

        charge_method = SUPPORTED_CHARGE_METHODS[partial_charge_method]

        if _cls is None:
            from openff.toolkit.topology.molecule import Molecule

            _cls = Molecule

        # Make a temporary copy of the molecule, since we'll be messing with its conformers
        mol_copy = _cls(molecule)

        if use_conformers is None:
            if charge_method["rec_confs"] == 0:
                mol_copy._conformers = None
            else:
                self.generate_conformers(
                    mol_copy,
                    n_conformers=charge_method["rec_confs"],
                    rms_cutoff=0.25 * unit.angstrom,
                )
                # TODO: What's a "best practice" RMS cutoff to use here?
        else:
            mol_copy._conformers = None
            for conformer in use_conformers:
                mol_copy._add_conformer(conformer)

        oemol = mol_copy.to_openeye()

        errfs = oechem.oeosstream()
        oechem.OEThrow.SetOutputStream(errfs)
        oechem.OEThrow.Clear()

        # The OpenFF toolkit has always supported a version of AM1BCC with no geometry optimization
        # or symmetry correction. So we include this keyword to provide a special configuration of quacpac
        # if requested.
        if partial_charge_method == "am1bccnosymspt":
            optimize = False
            symmetrize = False
            quacpac_status = oequacpac.OEAssignCharges(
                oemol, charge_method["oe_charge_method"](optimize, symmetrize)
            )
        elif partial_charge_method == "am1bccnoopt":
            optimize = False
            symmetrize = True
            quacpac_status = oequacpac.OEAssignCharges(
                oemol, charge_method["oe_charge_method"](optimize, symmetrize)
            )
        else:
            oe_charge_method = charge_method["oe_charge_method"]

            if callable(oe_charge_method):
                oe_charge_method = oe_charge_method()

            quacpac_status = oequacpac.OEAssignCharges(oemol, oe_charge_method)

        oechem.OEThrow.SetOutputStream(oechem.oeerr)  # restoring to original state
        # This logic handles errors encountered in #34, which can occur when using ELF10 conformer selection
        if not quacpac_status:

            oe_charge_engine = (
                oequacpac.OEAM1Charges
                if partial_charge_method == "am1elf10"
                else oequacpac.OEAM1BCCCharges
            )

            if "SelectElfPop: issue with removing trans COOH conformers" in (
                errfs.str().decode("UTF-8")
            ):
                print(
                    f"Warning: charge assignment involving ELF10 conformer selection failed due to a known bug (toolkit issue "
                    f"#346). Downgrading to {oe_charge_engine.__name__} charge assignment for this molecule. More information"
                    f"is available at https://github.com/openforcefield/openff-toolkit/issues/346"
                )
                quacpac_status = oequacpac.OEAssignCharges(oemol, oe_charge_engine())

        if quacpac_status is False:
            print(
                f'Unable to assign charges: {errfs.str().decode("UTF-8")}'
            )

        # Extract and return charges
        ## TODO: Make sure atom mapping remains constant

        charges = unit.Quantity(
            np.zeros(shape=oemol.NumAtoms(), dtype=np.float64), unit.elementary_charge
        )
        for oeatom in oemol.GetAtoms():
            index = oeatom.GetIdx()
            charge = oeatom.GetPartialCharge()
            charge = charge * unit.elementary_charge
            charges[index] = charge

        molecule.partial_charges = charges
class ProgressBar():
    # just discovered \r, so I thought I could try making a progress bar
    def __init__(self, iterable, bar_len=50):
        # manually count the items. This could potentially be very slow!
        c = 0
        for i in iterable:
            c += 1
        self.count = c
        self.curr = 0
        self.bar_len = bar_len
        self.iterable = iterable
        self.total_time = 0
    
    def progress(self):
        # return the number filled bars and empty bars
        percent_done = float(self.curr) / float(self.count)
        filled = percent_done * float(self.bar_len)
        filled = int(filled)
        empty = self.bar_len - filled
        return filled
    
    def printbar(self, filled):
        if self.curr != 0:
            time_per_iter = self.total_time / self.curr
        else:
            time_per_iter = 0
        time_left = (self.count - self.curr) * time_per_iter
        if time_left == 0 and self.curr != 0:
            time_str = "DONE"
        else:
            time_str = str(datetime.timedelta(seconds=time_left))
        empty = self.bar_len - filled
        print("\rPROGRESS: [", end="")
        print(bytes((219,)).decode('cp437') * filled, end="")
        print(" " * empty, end="")
        print(f"] ({self.curr}/{self.count}) ({time_str})", end="")
        sys.stdout.write("\033[K")
        print()

    def __iter__(self):

        for i in self.iterable:
            # update the progress bar
            filled = self.progress()
            self.printbar(filled)
            self.curr += 1
            # timing function will be usefull
            before = time()
            yield i
            after = time()
            delta_time = after - before
            self.total_time += delta_time

        self.printbar(self.bar_len)
        print()

if __name__ == "__main__":
    os.chdir("am1_bcc_scheme")
    assert(Path.cwd() == Path('/home/coda3831/openff-workspace/am1_bcc_scheme'))

    # methods = ["original",
    #             "maxcyc_0",
    #             "shake",
    #             "smirnoff",
    #             "openeye"]
    # output_dir_str = 'solution_testing'
    # delim = ","
    # columns = methods + ["conf", "isomorphic"]
    # df = pd.DataFrame(columns=columns)

    # 
    # for method in methods:
    #     with open(f"{output_dir_str}/{method}_{output_dir_str}_status_results.txt","r") as file:
    #         lines = file.read().split('\n')
    #     for line in lines:
    #         info = line.split(delim)
    #         name = info[0]
    #         if name == "":
    #             break
    #         conf = info[1]
    #         conf = int(conf)
    #         info = info[2:]
    #         if method == "openeye":
    #             info = [i[:-2] for i in info]
            
    #         charges = [float(i) for i in info if (i != "True" and i !="False")]
            
    #         df.at[name, method] = charges
    #         df.at[name, "conf"] = conf
    #         if method == "original":
    #             status = [(i=="True") for i in info if (i == "True" or i =="False")]
    #             status = any(status)
    #             df.at[name, "isomorphic"] = status
    # # due to alignment error, will fix later
    # df.dropna(inplace=True)

    # bad_mols = df[df['isomorphic'] == False]
    # ____________________  sqm connectivity change testing ___________________________
    # running this with rdkit version 2020.09.5
    # output_dir_str = 'final_connectivity_test'
    # delim = ","
    # open(f"{output_dir_str}_status_results.txt","w").write(f"name{delim}conf{delim}rdkit{delim}qcelemental\n")
    
    # for mol, fetch_status in ProgressBar(MolDatabase(only_smiles=False)):
    #     try:
    #         output_dir = Path.cwd() / output_dir_str
    #         net_charge = int(mol.total_charge / unit.elementary_charge)
            
    #         conformer_method(mol, method="openeye")

    #         args = ["-ek", 
    #                 "qm_theory='AM1', grms_tol=0.0005, scfconv=1.d-10, ndiis_attempts=700, qmcharge={net_charge}, maxcyc=2000"]
    #         status = run_am1_bcc(mol, output_dir=f"{output_dir_str}/{mol.name}", arguments=args)

    #         mol_dir = output_dir / f"{mol.name}"
    #         line = ""
    #         failed_confs = []
    #         for idx, row in status.iterrows():
    #             line += (mol.name + delim + str(idx) + delim)
    #             for item in row:
    #                 if item == False:
    #                     failed_confs.append(idx)
    #                 line += (str(item) + delim)
    #             line += "\n"
    #         failed_confs = list(set(failed_confs))

    #         open(f"{output_dir_str}_status_results.txt","a").write(line)

    #         # if failed_confs:
    #         #     # generate the animation set of there are any conformers that have failed 
    #         #     generage_simulation_frames_sqm(mol, output_dir=f"{output_dir_str}/{mol.name}", step_mult=1, select_confs=failed_confs)
    #     except Exception as e:
    #         print("major exception in outer loop")
    #         print(e)


    # MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
    # ______________animation generation _________________
    #  -the following lines were used to make animations of 
    #   unrestrained ambertools am1 simulations
    # WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
    # 
    # output_dir_str = 'final_connectivity_test'
    # output_dir_str = 'solution_testing'
    # delim = ","
    # mols = {}
    # with open(f"solution_testing/original_solution_testing_status_results.txt", "r") as file:
    #     lines = file.read().split("\n")
    # for line in lines:
    #     if line == '':
    #         continue
    #     info = line.split(",")
    #     if info[2] == "True" and info[3] == "True":
    #         if info[0] in mols.keys():
    #             mols[info[0]].append(int(info[1]))
    #         else:
    #             mols[info[0]] = [int(info[1])]
    
    # for mol, fetch_status in ProgressBar(MolDatabase(only_smiles=False, select_mols=list(mols.keys()), task=task_number, total_tasks=total_tasks)):
    #     try:
    #         output_dir = Path.cwd() / output_dir_str
            
    #         conformer_method(mol) # this needs to be changed, so everything needs to be changed. 

    #         failed_confs = mols[mol.name]

    #         mol_dir = output_dir / f"{mol.name}" / "original"


    #         generage_simulation_frames_sqm(mol, output_dir=str(mol_dir), step_mult=1, select_confs=failed_confs)
    #     except Exception as e:
    #         print("major exception in outer loop")
    #         print(e)

    # MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
    # _________________________solution testing _____________________________
    # -tests a number of solutions to attempt to restrain ambertools am1 simulations 
    #  and output their charges
    # -openeye charges are also generated here for comparison 
    # WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
    output_dir_str = 'solution_testing'
    delim = ","
    output_dir = Path.cwd() / output_dir_str
    methods = ["original",
                "maxcyc_0",
                "shake",
                "smirnoff",
                "openeye"]
    # if task_number == 0: # this will not lead to problems down the line... definitely not 
    #     Path(output_dir).mkdir(parents=True, exist_ok = True)
    #     # create output files with no header
    #     for method in methods:
    #         open(f"{output_dir_str}/{method}_{output_dir_str}_status_results.txt","w").write(f"")


    for mol, fetch_status in ProgressBar(MolDatabase(only_smiles=False, task=task_number, total_tasks=total_tasks)):
        try:
            # here we implement four methods, save the resulting charges, and output to the respective text file
            conformer_method(mol)
            net_charge = int(mol.total_charge / unit.elementary_charge)

            #--------------------------------------------------------------------------------
            # METHOD 1 ----------------------------------------------------------------------
            #--------------------------------------------------------------------------------
            # ORIGINAL
            args = ["-ek", 
                    f"qm_theory='AM1', grms_tol=0.0005, scfconv=1.d-10, ndiis_attempts=700, qmcharge={net_charge}, maxcyc=2000"]
            status = run_am1_bcc(mol, output_dir=f"{output_dir_str}/{mol.name}/original", arguments=args)

            line = ""
            for conf, i in zip(mol.conformers, range(0, len(mol.conformers))):
                mol_charges = output_dir / f"{mol.name}" / "original" / f"conf{i}" / f"charges_{mol.name}.txt"
                with open(str(mol_charges)) as file:
                    charges = file.read().split()
                line += (mol.name + delim)
                line += (f"{i}" + delim)
                for item in status.iloc[i]:
                    line += (str(item) + delim)
                line += delim.join(charges)
                line += "\n"

            open(f"{output_dir_str}/original_{output_dir_str}_status_results.txt","a").write(line) # hopefully this operation will be atomic? Can lock file ops later if needed

            #--------------------------------------------------------------------------------
            # METHOD 2 ----------------------------------------------------------------------
            #--------------------------------------------------------------------------------
            # MAXCYC = 0 (most promising)

            args = ["-ek", 
                    f"qm_theory='AM1', grms_tol=0.0005, scfconv=1.d-10, ndiis_attempts=700, qmcharge={net_charge}, maxcyc=0"]
            status = run_am1_bcc(mol, output_dir=f"{output_dir_str}/{mol.name}/maxcyc_0", arguments=args)

            line = ""
            for conf, i in zip(mol.conformers, range(0, len(mol.conformers))):
                mol_charges = output_dir / f"{mol.name}" / "maxcyc_0" / f"conf{i}" / f"charges_{mol.name}.txt"
                with open(str(mol_charges)) as file:
                    charges = file.read().split()
                line += (mol.name + delim)
                line += (f"{i}" + delim)
                for item in status.iloc[i]:
                    line += (str(item) + delim)
                line += delim.join(charges)
                line += "\n"

            open(f"{output_dir_str}/maxcyc_0_{output_dir_str}_status_results.txt","a").write(line)

            # #--------------------------------------------------------------------------------
            # # METHOD 3 ----------------------------------------------------------------------
            # #--------------------------------------------------------------------------------
            # # QMSHAKE (preliminary tests show that this does not do much to prevent large geometry changes)
            # args = ["-ek", 
            #         f"qm_theory='AM1', grms_tol=0.0005, scfconv=1.d-10, ndiis_attempts=700, qmcharge={net_charge}, maxcyc=2000, qmshake=1"]
            # status = run_am1_bcc(mol, output_dir=f"{output_dir_str}/{mol.name}/shake", arguments=args)

            # line = ""
            # for conf, i in zip(mol.conformers, range(0, len(mol.conformers))):
            #     mol_charges = output_dir / f"{mol.name}" / "shake" / f"conf{i}" / f"charges_{mol.name}.txt"
            #     with open(str(mol_charges)) as file:
            #         charges = file.read().split()
            #     line += (mol.name + delim)
            #     line += (f"{i}" + delim)
            #     for item in status.iloc[i]:
            #         line += (str(item) + delim)
            #     line += delim.join(charges)
            #     line += "\n"

            # open(f"{output_dir_str}/shake_{output_dir_str}_status_results.txt","a").write(line)
            # #--------------------------------------------------------------------------------
            # # METHOD 4 ----------------------------------------------------------------------
            # #--------------------------------------------------------------------------------
            # # SMIRNOFF (successful but computational cost does not justify the marginal (if even
            # # noticeable improvement over maxcyc=0). Not pursued further)

            # mol_copy = Molecule(mol)
            # conf_num = 0
            # min_conformers = []
            # for conf in mol.conformers:
            #     with tempfile.TemporaryDirectory() as tmpdir:
            #         with temporary_cd(tmpdir):
            #             mol_conf = Molecule(mol)
            #             mol_conf._conformers = [conf]
            #             print(type(conf._value))
            #             mol_conf.to_file(file_path='mol.pdb', file_format="PDB")
            #             pdbfile = PDBFile('mol.pdb')
            #             omm_topology = pdbfile.topology

            #             off_topology = mol_conf.to_topology()
            #             forcefield = ForceField('openff_unconstrained-1.3.0.offxml')
            #             system = forcefield.create_openmm_system(off_topology)
            #             time_step = 2*unit.femtoseconds  # simulation timestep
            #             temperature = 3000*unit.kelvin  # simulation temperature
            #             friction = 1/unit.picosecond  # collision rate
            #             integrator = openmm.LangevinIntegrator(temperature, friction, time_step)
            #             simulation = openmm.app.Simulation(omm_topology, system, integrator)
            #             positions = pdbfile.getPositions() 
            #             simulation.context.setPositions(positions)

            #             pdb_reporter = openmm.app.PDBReporter('trajectory.pdb', 10)
            #             simulation.reporters.append(pdb_reporter)
                        
            #             simulation.minimizeEnergy(maxIterations=1000)
            #             st = simulation.context.getState(getPositions=True, getEnergy=True)
            #             print(st.getPotentialEnergy())
            #             print(st.getPositions())
            #             unitless_positions = []
            #             for vec in st.getPositions():
            #                 x = vec.x * 10     # please don't let me forget this
            #                 y = vec.y * 10     # how to do this in a... better... way 
            #                 z = vec.z * 10
            #                 unitless_positions.append([x, y, z])
            #             unitless_positions = numpy.array(unitless_positions)
            #             final_conf = unit.Quantity(unitless_positions, mol_conf._conformers[0].unit) # units should be angstrom
            #             min_conformers.append(final_conf)

            # mol_copy._conformers = min_conformers
            # args = ["-ek", 
            #         f"qm_theory='AM1', grms_tol=0.0005, scfconv=1.d-10, ndiis_attempts=700, qmcharge={net_charge}, maxcyc=0"]
            # status = run_am1_bcc(mol_copy, output_dir=f"{output_dir_str}/{mol.name}/smirnoff", arguments=args)

            # line = ""
            # for conf, i in zip(mol_copy.conformers, range(0, len(mol_copy.conformers))):
            #     mol_charges = output_dir / f"{mol.name}" / "smirnoff" / f"conf{i}" / f"charges_{mol_copy.name}.txt"
            #     with open(str(mol_charges)) as file:
            #         charges = file.read().split()
            #     line += (mol_copy.name + delim)
            #     line += (f"{i}" + delim)
            #     for item in status.iloc[i]:
            #         line += (str(item) + delim)
            #     line += delim.join(charges)
            #     line += "\n"

            # open(f"{output_dir_str}/smirnoff_{output_dir_str}_status_results.txt","a").write(line)


            #--------------------------------------------------------------------------------
            # METHOD 5 ----------------------------------------------------------------------
            #--------------------------------------------------------------------------------
            # OPENEYE, no frills or major modifications 
            mol_copy = Molecule(mol)
            line = ""
            for conf, i in zip(mol.conformers, range(0, len(mol.conformers))):
                mol_conf = Molecule(mol)
                mol_conf._conformers = [conf]
                mol_conf.assign_partial_charges(partial_charge_method='am1bcc', 
                                                use_conformers=[conf], 
                                                toolkit_registry=OpenEyeToolkitWrapper())
                charges = [str(i) for i in mol_conf.partial_charges]
                line += (mol_copy.name + delim)
                line += (f"{i}" + delim)
                line += delim.join(charges)
                line += "\n"

            open(f"{output_dir_str}/openeye_{output_dir_str}_status_results.txt","a").write(line)
            # #--------------------------------------------------------------------------------
            # # METHOD 6 ----------------------------------------------------------------------
            # #--------------------------------------------------------------------------------
            # # GEOMETRIC (hard to make work)
            # # NOTE: this approach did not pan out as geometric would error often
            # line = ""
            # for conf, i in zip(mol.conformers, range(0, len(mol.conformers))):
            #     off_molecule = Molecule(mol)
            #     off_molecule._conformers = [conf]
            #     frame_output_dir_str = f"{output_dir_str}/{mol.name}/geo_minor/conf{i}"
            #     new_dir = Path(Path.cwd() / frame_output_dir_str)
            #     new_dir.mkdir(exist_ok=True, parents=True)

            #     file = f"{output_dir_str}/{mol.name}/original/conf{i}/sqm_{mol.name}.pdb"
            #     rdmol = Chem.MolFromPDBFile(file, removeHs=False)

            #     geo_molecule = GeoMolecule()
            #     geo_molecule.Data = {
            #         "resname": ["UNK"] * off_molecule.n_atoms,
            #         "resid": [0] * off_molecule.n_atoms,
            #         "elem": [atom.element.symbol for atom in off_molecule.atoms],
            #         "bonds": [
            #             (bond.atom1_index, bond.atom2_index) for bond in off_molecule.bonds
            #         ],
            #         "xyzs": [
            #             conformer.value_in_unit(unit.angstrom) 
            #             for conformer in off_molecule.conformers
            #         ],
            #     }
            #     #MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
            #     # APPLYING THE xyz constraint
            #     #WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
            #     constrained_idxs = []
            #     for rd_atom, off_atom in zip(rdmol.GetAtoms(), off_molecule.atoms):
            #         assert(rd_atom.GetAtomicNum() == off_atom.atomic_number)
            #         if len(rd_atom.GetBonds()) != len(off_atom.bonds):
            #             constrained_idxs.append(off_atom.molecule_atom_index)
            #             off_neighbors = []
            #             rd_neighbors = []
            #             for bond in off_atom.bonds:
            #                 off_neighbors.append(bond.atom1_index)
            #                 off_neighbors.append(bond.atom2_index)
            #             for bond in rd_atom.GetBonds():
            #                 rd_neighbors.append(bond.GetEndAtomIdx())
            #                 rd_neighbors.append(bond.GetBeginAtomIdx())
            #             off_neighbors = list(set(off_neighbors))
            #             rd_neighbors = list(set(rd_neighbors))
            #             if len(off_neighbors) > len(rd_neighbors):
            #                 l1 = list(numpy.setdiff1d(off_neighbors, rd_neighbors))
            #             else:
            #                 l1 = list(numpy.setdiff1d(rd_neighbors, off_neighbors))
            #             for l in l1:
            #                 constrained_idxs.append(l)
            #     constrained_idxs = sorted(list(set(constrained_idxs)))

                
                
            #     constraint_input = "$freeze\nxyz "
            #     inner_i = 0
            #     for idx in constrained_idxs:
            #         if inner_i == 0:
            #             constraint_input += f"{idx+1}"
            #         else:
            #             constraint_input += f",{idx+1}"
            #         inner_i += 1
            #     Cons, CVals = None, None
            #     Cons, CVals = ParseConstraints(geo_molecule, constraint_input)

            #     coord_sys = DelocalizedInternalCoordinates(
            #         geo_molecule,
            #         build=True,
            #         connect=True,
            #         addcart=False,
            #         constraints=Cons,
            #         cvals=CVals[0] if CVals is not None else None
            #     )
                
            #     try:
            #         result = Optimize(
            #             coords=geo_molecule.xyzs[0].flatten() * ang2bohr,
            #             molecule=geo_molecule,
            #             IC=coord_sys,
            #             engine=SQMAM1(geo_molecule, net_charge, save_output=True, frame_output_dir=frame_output_dir_str, mol_name=off_molecule.name, conf_idx=i),
            #             dirname=f"tmp-dir",
            #             params=OptParams(
            #                 convergence_energy=10.0,
            #                 convergence_grms=1.0,
            #                 convergence_gmax=10.0,
            #                 convergence_drms=10.0,
            #                 convergence_dmax=10.0,
            #                 maxiter=40
            #             )
            #         )
            #     except GeomOptNotConvergedError:
            #         pass

            #     with temporary_cd(frame_output_dir_str):
            #         select_frames = sorted(list(Path('charge_frames').glob('*')))[-4:]
            #         avg_charges = []
            #         for frame in select_frames:
            #             with open(frame, 'r') as ifile:
            #                 charges = ifile.read()
            #             charges = charges.split()
            #             charges = [float(charge) for charge in charges]
            #             avg_charges.append(charges)
            #     avg_charges = numpy.array(avg_charges)
            #     geo_charges = numpy.mean(avg_charges, axis=0)
            #     line += (mol.name + delim)
            #     line += (f"{i}" + delim)
            #     line += delim.join([str(a) for a in geo_charges])
            #     line += "\n"

            # open(f"{output_dir_str}/geo_minor_{output_dir_str}_status_results.txt","a").write(line)

            #--------------------------------------------------------------------------------
            # METHOD 7 ----------------------------------------------------------------------
            #--------------------------------------------------------------------------------
            # OPENEYE with maxcyc=0 constraint
            mol_copy = Molecule(mol)
            line = ""
            for conf, i in zip(mol.conformers, range(0, len(mol.conformers))):
                mol_conf = Molecule(mol)
                mol_conf._conformers = [conf]
                mol_conf.assign_partial_charges(partial_charge_method='am1bccnosymspt', 
                                                use_conformers=[conf], 
                                                toolkit_registry=OpenEyeToolkitWrapper())
                charges = [str(i) for i in mol_conf.partial_charges]
                line += (mol_copy.name + delim)
                line += (f"{i}" + delim)
                line += delim.join(charges)
                line += "\n"

            open(f"{output_dir_str}/openeye_nosymspt_{output_dir_str}_status_results.txt","a").write(line)
            #--------------------------------------------------------------------------------
            # METHOD 8 ----------------------------------------------------------------------
            #--------------------------------------------------------------------------------
            # OPENEYE with maxcyc=0 constraint (but letting symmetry be True)
            mol_copy = Molecule(mol)
            line = ""
            for conf, i in zip(mol.conformers, range(0, len(mol.conformers))):
                mol_conf = Molecule(mol)
                mol_conf._conformers = [conf]
                oe_toolkit_wrapper = OpenEyeToolkitWrapper()
                my_assign_partial_charges("bla", mol_conf, partial_charge_method='am1bccnoopt', 
                                                use_conformers=[conf])
                charges = [str(i) for i in mol_conf.partial_charges]
                line += (mol_copy.name + delim)
                line += (f"{i}" + delim)
                line += delim.join(charges)
                line += "\n"

            open(f"{output_dir_str}/openeye_noopt_{output_dir_str}_status_results.txt","a").write(line)

        except Exception as e:
            print("major exception in outer loop")
            print(e)