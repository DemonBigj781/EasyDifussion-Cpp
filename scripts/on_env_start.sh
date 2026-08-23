#!/bin/env bash

source ./scripts/functions.sh

printf "\n\nEasy Diffusion - v3\n\n"

export PYTHONNOUSERSITE=y

if [ -f "scripts/config.sh" ]; then
    source scripts/config.sh
fi

if [ -f "scripts/user_config.sh" ]; then
    source scripts/user_config.sh
fi

# setup environment
if [ -e "installer_files/env" ]; then
	export ENVFOLDER="$(pwd)/installer_files/env"
	export PATH="${ENVFOLDER}/bin:$PATH"; 
	# check python version and adjust PYTHONPATH
	if [ -e "${ENVFOLDER}/bin/python" ]; then
		export PYTHONVERSION="$(${ENVFOLDER}/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
		export PYTHONPATH="${ENVFOLDER}/lib/python${PYTHONVERSION}/site-packages"
	fi
fi

if [ ! -d "sd-ui-files/ui" ]; then
    fail "The local Easy Diffusion source checkout is missing: sd-ui-files/ui"
fi

# Deploy the checked-in local fork without fetching or resetting it.
rm -rf ui
cp -Rf sd-ui-files/ui .
cp sd-ui-files/scripts/on_sd_start.sh scripts/
cp sd-ui-files/scripts/bootstrap.sh scripts/
cp sd-ui-files/scripts/check_modules.py scripts/
cp sd-ui-files/scripts/get_config.py scripts/
cp sd-ui-files/scripts/config.yaml.sample scripts/
cp sd-ui-files/scripts/webui_console.py scripts/
cp sd-ui-files/scripts/start.sh .
cp sd-ui-files/scripts/developer_console.sh .
cp sd-ui-files/scripts/functions.sh scripts/

exec ./scripts/on_sd_start.sh
