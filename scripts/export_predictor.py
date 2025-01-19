import lightgbm as lgb
import tl2cgen
import treelite
import pickle   # if your predictor is pickled
import os
# if the predictor is pickled

for model_type in ["class_70_light", "class_75_light", "class_80_light", "class_85_light",
                   "low_model_5_light", "low_model_10_light", "median_model_light"]:

    with open(os.path.join(os.pardir, "data/models/", model_type) + '.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    # if the predictor is a LightGBM .txt model export
    params = {"parallel_comp": 16}  # this will split the resulting C export into 16 files for parallel compilation

    treelite_model = treelite.frontend.from_lightgbm(model)
    library_name = "EBG_" + model_type

    tl2cgen.export_srcpkg(
        treelite_model,
        toolchain="gcc",
        pkgpath=os.path.join(os.pardir, "data/models_c/", model_type) + ".zip",
        libname=library_name,
        params=params,
        verbose=False
    )