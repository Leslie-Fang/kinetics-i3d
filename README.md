# I3D models trained on UCF101
  Forked from: https://github.com/deepmind/kinetics-i3d.
  Refer to README.md.origin for the origin readme file

## Steps
  The origin model was trained with TF1.14 and sonnet(requirements.txt list the package version)
  Here we transfer it to TF2.0(although we still disable the eager model, we can train and inference it in TF2.0)

  ### Step1: Generate the new model
  #### Set the TF1.14 env
  ```
  pip install -r requirements.txt
  ```
  #### Generate the model for TF2.0
  ```
  python main.py
  ```
  The origin model was trained with Kinetics dataset support **RGB and flow input**.
  The origin model has 400 classes.

  In the main.py, we first load parts of the origin model and restore these parameters from the origin checkpoint.
  Then we add a new layer of Conv3D with 101 classes as output(ucf101 has 101 class).
  Then we save the models in fine_tune_model

  ### Step2: Midstep
  This steps is used for tunning code only.
  We write a main_tf2.py which should run in TF1.14.
  This is a mid-version which try restore models from fine_tune_model.
  We can change it directly to TF2.0 version (https://tensorflow.google.cn/guide/migrate#saving_loading).
  Run it in tf2.0 ENV, tf_upgrade_v2 is a build in tool.
  ```
  tf_upgrade_v2 --infile main_tf2.py --outfile main_tf2_convert.py
  ```

  ### Step3: train and inference in TF2.0
  In the main_tf2_convert.py, we define the function to train and inference.
  #### Set the TF2.0 env
  ```
  pip install -r requirements_tf2.txt
  ```



