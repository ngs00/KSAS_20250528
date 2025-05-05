import glob
import jcamp
import numpy
from rdkit.Chem import MolFromSmarts, MolFromSmiles
from scipy.interpolate import interp1d


def load_dataset(metadata, path_jdx_files, target_fg):
    jdx_files = glob.glob(path_jdx_files + '/*.jdx')
    target_fg = MolFromSmarts(target_fg)
    dataset_x = list()
    dataset_y = list()

    for jdx_file in jdx_files:
        irs = jcamp.jcamp_readfile(jdx_file)
        smiles = metadata[jdx_file.split('\\')[-1].split('.jdx')[0]][3]

        if isinstance(smiles, float):
            continue

        if 'path length' in irs.keys():
            continue

        if irs['xunits'] != '1/CM':
            continue

        # Convert transmittance to absorbance.
        if irs['yunits'] == 'TRANSMITTANCE':
            jcamp.jcamp_calc_xsec(irs, skip_nonquant=False)

        # Perform polynomial interpolation.
        wavenumber = irs['x']
        absorbance = irs['y']
        absorbance = numpy.nan_to_num(absorbance, nan=0)
        f_interpol = interp1d(wavenumber, absorbance, kind='linear', fill_value='extrapolate')
        wavenumber = numpy.arange(550, 4000 + 2, step=2)
        absorbance = f_interpol(wavenumber)

        # Calculate label data.
        mol = MolFromSmiles(smiles)
        label = 1 if mol.HasSubstructMatch(target_fg) else 0

        # Append training data.
        dataset_x.append(absorbance)
        dataset_y.append(label)

    return numpy.vstack(dataset_x), numpy.array(dataset_y)
