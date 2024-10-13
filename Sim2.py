import os
import numpy as np
import matplotlib.pyplot as plt
from lenstronomy.SimulationAPI.model_api import ModelAPI
from lenstronomy.SimulationAPI.sim_api import SimAPI
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from astropy.io import fits


def gen_fits(data, filename):
    data = data.astype(np.int8, copy=False)
    fits.ImageHDU(data=data).writeto(f'conjunto_entrenamiento/{filename}.fits', overwrite=True)

def generar_imagen_lente(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, tel_config):
    # Configuración de los datos de observación
    tel_config = tel_config.kwargs_single_band()


    # Definir modelos de lentes y fuentes
    # Single-Plane para lente y fuente. perfil de sersic elíptico
    kwargs_model = {'lens_model_list': ['SIE', 'SHEAR'],  # list of lens models to be used
                    'lens_light_model_list': ['SERSIC_ELLIPSE'],  # list of unlensed light models to be used
                    'source_light_model_list': ['SERSIC_ELLIPSE'],  # list of extended source models to be used
                    'point_source_model_list': ['SOURCE_POSITION']  # list of point source models to be used
                    }

    # Simulación
    sim = SimAPI(numpix=512, kwargs_single_band=tel_config, kwargs_model=kwargs_model)
    gen = sim.image_model_class()

    kwargs_lens_light, kwargs_source, kwargs_ps = sim.magnitude2amplitude(kwargs_lens_light, kwargs_source, kwargs_ps)

    image = gen.image(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps)

    return image


from JST250 import JST250


def gen_JST_bands():
    jst = [
        JST250(band='gSDSS', psf_type='GAUSSIAN', coadd_years=3),
        JST250(band='rSDSS', psf_type='GAUSSIAN', coadd_years=3),
        JST250(band='iSDSS', psf_type='GAUSSIAN', coadd_years=3),
        JST250(band='narrow', psf_type='GAUSSIAN', coadd_years=3)
    ]

    return jst


# Directorio donde se guardarán las imágenes
output_dir = 'conjunto_entrenamiento'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Número de imágenes a generar
num_images = 1
fov = 50 # arcsec

# Configuración de los parámetros de las lentes y fuentes
for i in range(num_images):
    # Definir parámetros aleatorios para la lente

    kwargs_lens = [
        {'theta_E': np.random.uniform(1, 2.5), # Radio de Einstein
         'e1': np.random.uniform(-0.4, 0.4), # Excentricidad 1
         'e2': np.random.uniform(-0.4, 0.4), # Excentricidad 2
         'center_x': np.random.uniform(-fov/2, fov/2),
         'center_y': np.random.uniform(-fov/2, fov/2)
         },
        # SIE model
        {'gamma1': np.random.uniform(-0.05, 0.05),
        'gamma2': np.random.uniform(-0.05, 0.05),
         'ra_0': 0,
         'dec_0': 0}
        # SHEAR model
    ]

    # Configurar la luz de la lente con variaciones aleatorias
    kwargs_lens_light = [{
        'magnitude': np.random.uniform(12, 18),  # Magnitud de la lente
        'R_sersic': np.random.uniform(0.3, 1.0),  # Radio de Sersic
        'n_sersic': np.random.uniform(2, 4),  # Índice de Sersic
        'e1': np.random.uniform(-0.3, 0.3),  # Excentricidad en la dirección 1
        'e2': np.random.uniform(-0.3, 0.3),  # Excentricidad en la dirección 2
        'center_x': np.random.uniform(-fov / 2, fov / 2),  # Centro en x
        'center_y': np.random.uniform(-fov / 2, fov / 2),  # Centro en y
    }]

    # Configurar la luz de la fuente con variaciones aleatorias
    kwargs_source = [{
        'magnitude': np.random.uniform(18, 23),  # Magnitud de la fuente
        'R_sersic': np.random.uniform(0.1, 0.5),  # Radio de Sersic
        'n_sersic': np.random.uniform(0.5, 2),  # Índice de Sersic
        'e1': np.random.uniform(-0.3, 0.3),  # Excentricidad en la dirección 1
        'e2': np.random.uniform(-0.3, 0.3),  # Excentricidad en la dirección 2
        'center_x': np.random.uniform(-fov / 2, fov / 2),  # Centro en x
        'center_y': np.random.uniform(-fov / 2, fov / 2),  # Centro en y
    }]

    # Fuente puntual
    kwargs_ps = [{
        'magnitude': np.random.uniform(20, 24),  # Magnitud de la fuente puntual
        'ra_source': np.random.uniform(-fov / 2, fov / 2),  # Posición en RA
        'dec_source': np.random.uniform(-fov / 2, fov / 2),  # Posición en DEC
    }]

    # Generar la imagen de la lente
    bands = gen_JST_bands()
    for band in bands:
        imagen = generar_imagen_lente(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, band)
        gen_fits(imagen, f'jst_{band.band}')

    if i % 100 == 0:
        print(f'{i} imágenes generadas')
