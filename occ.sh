set -x

DEVICE=${1:-"6,7"}
UTIL=${2:-"0.4"}

/data1/miniconda3/envs/imggs/bin/python main.py --device ${DEVICE} --util ${UTIL}