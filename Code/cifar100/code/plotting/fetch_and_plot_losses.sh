export GRANDPARENTNAME="$(dirname "$(dirname $PWD)")"
export NAME="${GRANDPARENTNAME##*/}"
mkdir ../../snapshots
scp fyp:~/Final-Year-Project/Code/$NAME/snapshots/net1_losses.npy ../../snapshots/
scp fyp:~/Final-Year-Project/Code/$NAME/snapshots/net2_losses.npy ../../snapshots/
scp fyp:~/Final-Year-Project/Code/$NAME/snapshots/net12_losses.npy ../../snapshots/
python plotting_losses.py
