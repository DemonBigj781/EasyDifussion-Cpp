#!/bin/bash

cp sd-ui-files/scripts/functions.sh scripts/
cp sd-ui-files/scripts/on_env_start.sh scripts/
cp sd-ui-files/scripts/bootstrap.sh scripts/
cp sd-ui-files/scripts/check_modules.py scripts/
cp sd-ui-files/scripts/get_config.py scripts/
cp sd-ui-files/scripts/config.yaml.sample scripts/
cp sd-ui-files/scripts/webui_console.py scripts/
cp sd-ui-files/scripts/ensure_torchruntime.py scripts/


source ./scripts/functions.sh

# activate the installer env
export CONDA_BASEPATH=$(conda info --base)
source "$CONDA_BASEPATH/etc/profile.d/conda.sh" # avoids the 'shell not initialized' error

conda activate || fail "Failed to activate conda"

# hack to fix conda 4.14 on older installations
cp $CONDA_BASEPATH/condabin/conda $CONDA_BASEPATH/bin/conda

# remove the old version of the dev console script, if it's still present
if [ -e "open_dev_console.sh" ]; then
    rm "open_dev_console.sh"
fi

if [ -e "ui/plugins/ui/merge.plugin.js" ]; then
    rm "ui/plugins/ui/merge.plugin.js"
fi

# use the current installer environment
if [ -e "installer_files/env" ]; then
    export INSTALL_ENV_DIR="$(pwd)/installer_files/env"
fi

# this is outside check_modules.py to ensure that the required version of torchruntime is present (installing if necessary)
python scripts/ensure_torchruntime.py

# Download the required packages
python scripts/check_modules.py --launch-uvicorn

read -p "Press any key to continue"
