from schnetpack import Properties
from schnetpack.md.calculators import MDCalculator
from schnetpack.md.utils import MDUnits
from schnetpack.environment import SimpleEnvironmentProvider, collect_atom_triples
import numpy as np
import torch

class SchnetPackCalculator(MDCalculator):
    """
    MD calculator for schnetpack models.

    Args:
        model (object): Loaded schnetpack model.
        required_properties (list): List of properties to be computed by the calculator
        force_handle (str): String indicating the entry corresponding to the molecular forces
        position_conversion (float): Unit conversion for the length used in the model computing all properties. E.g. if
                             the model needs Angstrom, one has to provide the conversion factor converting from the
                             atomic units used internally (Bohr) to Angstrom: 0.529177...
        force_conversion (float): Conversion factor converting the forces returned by the used model back to atomic
                                  units (Hartree/Bohr).
        property_conversion (dict(float)): Optional dictionary of conversion factors for other properties predicted by
                                           the model. Only changes the units used for logging the various outputs.
        detach (bool): Detach property computation graph after every calculator call. Enabled by default. Should only
                       be disabled if one wants to e.g. compute derivatives over short trajectory snippets.
    """

    def __init__(
        self,
        model,
        atoms,
        required_properties,
        force_handle,
        position_conversion=1.0 / MDUnits.angs2bohr,
        force_conversion=1.0 / MDUnits.auforces2aseforces,
        property_conversion={},
        detach=True,
        collect_triples=True,
    ):
        super(SchnetPackCalculator, self).__init__(
            required_properties,
            force_handle,
            position_conversion,
            force_conversion,
            property_conversion,
            detach,
        )
        
        self.atoms = atoms
        self.model = model
        self.collect_triples = collect_triples
        
    def calculate(self, system):
        """
        Main routine, generates a properly formatted input for the schnetpack model from the system, performs the
        computation and uses the results to update the system state.

        Args:
            system (schnetpack.md.System): System object containing current state of the simulation.
        """
        inputs = self._generate_input(system)
        self.results = self.model(inputs)
        self._update_system(system)

    def _generate_input(self, system):
        """
        Function to extracts neighbor lists, atom_types, positions e.t.c. from the system and generate a properly
        formatted input for the schnetpack model.

        Args:
            system (schnetpack.md.System): System object containing current state of the simulation.

        Returns:
            dict(torch.Tensor): Schnetpack inputs in dictionary format.
        """

        # If requested get neighbor lists for triples
        inputs = self._convert_atoms()

        if self.collect_triples:
            mask_triples = torch.ones_like(inputs[Properties.neighbor_pairs_j])
            mask_triples[inputs[Properties.neighbor_pairs_j] < 0] = 0
            mask_triples[inputs[Properties.neighbor_pairs_k] < 0] = 0
            inputs[Properties.neighbor_pairs_mask] = mask_triples.float()
            for key, value in inputs.items():
                inputs[key] = value.unsqueeze(0)

        positions, atom_types, atom_masks = self._get_system_molecules(system)
        neighbors, neighbor_mask = self._get_system_neighbors(system)

        inputs[Properties.R] = positions
        inputs[Properties.Z] = atom_types
        inputs[Properties.atom_mask] = atom_masks
        inputs[Properties.cell] = None
        inputs[Properties.cell_offset] = None
        inputs[Properties.neighbors] = neighbors
        inputs[Properties.neighbor_mask] = neighbor_mask

        return inputs

    def _convert_atoms(
        self,
        centering_function=None,
        output=None,
    ):
        """
            Helper function to convert ASE atoms object to SchNetPack input format.
            Args:
                atoms (ase.Atoms): Atoms object of molecule
                environment_provider (callable): Neighbor list provider.
                collect_triples (bool, optional): Set to True if angular features are needed.
                centering_function (callable or None): Function for calculating center of
                    molecule (center of mass/geometry/...). Center will be subtracted from
                    positions.
                output (dict): Destination for converted atoms, if not None
        Returns:
            dict of torch.Tensor: Properties including neighbor lists and masks
                reformated into SchNetPack input format.
        """
        if output is None:
            inputs = {}
        else:
            inputs = output


        # get atom environment
        environment_provider=SimpleEnvironmentProvider()
        nbh_idx, offsets = environment_provider.get_environment(self.atoms)



        # If requested get neighbor lists for triples
        if self.collect_triples:
            nbh_idx_j, nbh_idx_k, offset_idx_j, offset_idx_k = collect_atom_triples(nbh_idx)
            inputs[Properties.neighbor_pairs_j] = torch.LongTensor(nbh_idx_j.astype(np.int))
            inputs[Properties.neighbor_pairs_k] = torch.LongTensor(nbh_idx_k.astype(np.int))

            inputs[Properties.neighbor_offsets_j] = torch.LongTensor(
                offset_idx_j.astype(np.int)
            )
            inputs[Properties.neighbor_offsets_k] = torch.LongTensor(
                offset_idx_k.astype(np.int)
            )

        return inputs
