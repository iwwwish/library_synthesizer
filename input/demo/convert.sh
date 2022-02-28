for file in R*/*.mrv
do
    /Users/siramshettyv2/work/software/jchem/bin/molconvert mol "$file" -o "${file%%.*}.mol"
done