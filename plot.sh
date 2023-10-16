help()
{
    echo
    echo "Plot 2D heat"
    echo
    echo "Syntax"
    echo "---------------------------------------------"
    echo "./plot.sh [-y|x|h]                           "
    echo
    echo "Option    Description     Arguments   Default"
    echo "---------------------------------------------"
    echo "y         x size          Optional    256    "
    echo "x         y size          Optional    256    "
    echo "h         Help            None               "
    echo
    echo "Example"
    echo "---------------------------------------------"
    echo "./plot.sh -y 256 -x 256                      "
    echo
}

#-----------------------------------------------------------------
set -e

y_size=256
x_size=256

while getopts ":y:x:h" opt; do
    case $opt in
        y)
            y_size=$OPTARG;;
        x)
            x_size=$OPTARG;;
        h)
            help
            exit;;
        \?)
            echo "Invalid option"
            help
            exit;;
    esac
done

#-----------------------------------------------------------------
SIZE_Y=`echo $y_size | bc`
SIZE_X=`echo $x_size | bc`

# Go through every data/#####.bin file and convert to image
for data in `find data/*.bin`
do
    image=`echo $data | sed s/data/plots/ | sed s/\.bin/.png/`
    echo "Plotting '$image'"
    cat <<END_OF_SCRIPT | gnuplot - &
set term png
set view map
set cbrange [0:60]
set output "$image"
plot "$data" binary array=${SIZE_Y}x${SIZE_X} format="%lf" with image
END_OF_SCRIPT
done
