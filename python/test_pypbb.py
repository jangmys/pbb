import pypbb
# import fsp_heuristics

# print(test_add.adder(2,3))

inst=pypbb.instance_taillard('ta102')
bound=pypbb.bound_fsp()
bound.init(inst)

s=pypbb.subproblem(200,[55,183,170,10,24,162,53,168,49,117,148,131,59,90,18,93,178,150,118,154,91,44,175,12,48,25,56,73,31,167,47,127,15,27,112,158,177,172,137,30,104,196,51,155,145,110,57,95,14,121,46,143,133,29,181,102,151,68,37,184,149,4,76,115,182,186,99,173,87,180,7,42,83,161,40,169,107,58,192,3,22,114,106,19,16,147,79,77,103,34,156,185,165,69,28,119,174,135,111,45,96,64,113,92,23,8,65,50,191,194,41,141,94,128,38,134,153,187,13,189,166,67,136,138,197,32,129,188,33,70,75,157,98,9,160,146,61,142,74,123,0,171,176,35,120,97,63,71,190,5,124,1,109,159,20,54,62,101,78,84,130,199,26,163,122,100,21,198,80,39,140,125,89,164,60,85,82,17,126,116,195,108,144,72,6,36,152,11,139,81,105,193,132,86,88,52,66,2,43,179])

for i in range(s.size):
    s.schedule[i]=s.schedule[i]-1
    print(i,s.schedule[i])

print(s)
print(bound.eval(s.schedule))




#
inst=pypbb.instance_taillard('ta20')
#


print(inst.get_job_number(20))
print(inst.get_machine_number(20))


bound=pypbb.bound_fsp()
bound.init(inst)

s=pypbb.subproblem(10)
print(s)

s=pypbb.subproblem(20,[1,3,5,7,9,11,13,15,17,19,0,2,4,6,8,10,12,14,16,18])
print(s)

print(bound.eval(s.schedule))





print('test NEH')

neh=pypbb.fastNEH(inst)
neh.run(s)
print(s)

p=neh.run()
print(p)

# ====================
print('test BB')

#static
pypbb.args.problem='f'
pypbb.args.inst_name='ta20'
pypbb.args.threads=4

inst=pypbb.instance_taillard(pypbb.args.inst_name)

# bb = pypbb.pbab()
bb = pypbb.pbab(inst)


# ivmbb = pypbb.make_ivmbb(bb)
#
# ivmbb.set_root([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]);
# ivmbb.init_at_interval([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0]);
# ivmbb.run();
#
# bb.print_stats()

nb_threads=4

pypbb.args.ws='a'

mcbb = pypbb.IVMController(bb,pypbb.args.threads)

mcbb.init_intervals();
mcbb.run()

bb.print_stats()
