#!/bin/bash

nodes=$1
pernode=4
threads=8

make_machinefile() {
    uniq $OAR_NODEFILE > my_machinefile_${nodes}

    sed -e "s/$/ slots=$1/" -i my_machinefile_${nodes}

    nplusone=$(( $1 + 1 ))

    sed -i "1!b;s/slots=$1/slots=${nplusone}/" my_machinefile_${nodes}
}

cd $HOME/permutationbb/build

make_machinefile ${pernode}

cmake -DCMAKE_BUILD_TYPE=Release -DMPI=true ..
make -j8

np=$(( ${nodes} * ${pernode} + 1 ))

outputdir=../output

mkdir -p ${outputdir}

for run in {1..10}
do
	for inst in 30 22 27 26 #24
	do
    	mpirun --mca pml ^ucx -np ${np} -hostfile my_machinefile_${nodes} ./distributed/dbb -z p=fsp,i=ta${inst},o -t ${threads} --bound 1 --primary-bound s --branch -2 > ${outputdir}/ta${inst}-n${nodes}-LB1-r${run} 2>&1

    	mpirun --mca pml ^ucx -np ${np} -hostfile my_machinefile_${nodes} ./distributed/dbb -z p=fsp,i=ta${inst},o -t ${threads} --bound 1 --primary-bound j --branch -2 > ${outputdir}/ta${inst}-n${nodes}-LB2-r${run} 2>&1

    	mpirun --mca pml ^ucx -np ${np} -hostfile my_machinefile_${nodes} ./distributed/dbb -z p=fsp,i=ta${inst},o -t ${threads} --bound 0 --branch -2 > ${outputdir}/ta${inst}-n${nodes}-LBDelta-r${run} 2>&1
	done
done
