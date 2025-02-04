####################################################################################
###
### vecno-miner
### https://github.com/tmrlvi/vecno-miner/releases
###
### Hive integration: Merlin
###
####################################################################################

#!/usr/bin/env bash
[[ -e /hive/custom ]] && . /hive/custom/vecno-miner/h-manifest.conf
[[ -e /hive/miners/custom ]] && . /hive/miners/custom/vecno-miner/h-manifest.conf
conf=""
conf+=" --vecno-address=$CUSTOM_URL --mining-address $CUSTOM_TEMPLATE"


[[ ! -z $CUSTOM_USER_CONFIG ]] && conf+=" $CUSTOM_USER_CONFIG"

echo "$conf"
echo "$conf" > $CUSTOM_CONFIG_FILENAME

