mpirun --mca orte_rsh_agent "oarsh" -hostfile `uniq $OAR_NODEFILE` -map-by ppr:2:node ./distributed/dbb -f ../bbconfig.ini -z p=fsp,i=$1
