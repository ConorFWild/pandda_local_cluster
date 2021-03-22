# pandda_local_cluster


## Installation:
 - git clone https://github.com/ConorFWild/pandda_local_cluster.git
 - cd pandda_autobuild
 - pip install .

## Running a single dataset:
 - Make sure rhofit is on your path
 - Make sure phenix is on your path
 - python pandda_autobuild/autobuild/autobuild.py model xmap mtz smiles x y z out_dir

## Running an entire PanDDA:
 - Make sure rhofit is on your path
 - Make sure phenix is on your path
 - python pandda_autobuild/autobuild/autobuild_pandda.py pandda_event_table out_dir
