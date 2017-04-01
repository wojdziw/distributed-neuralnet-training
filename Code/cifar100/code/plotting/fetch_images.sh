export GRANDPARENTNAME="$(dirname "$(dirname $PWD)")"
export NAME="${GRANDPARENTNAME##*/}"
mkdir ../../images
scp fyp:~/Final-Year-Project/Code/$NAME/images/*.png ../../images
