import pypbb
# import fsp_heuristics

# print(test_add.adder(2,3))
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
pypbb.args.inst_name='ta30'
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

mcbb.set_ws(pypbb.make_victim_selector(pypbb.args.threads,'a'))
mcbb.init_intervals(1,[0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0]);
mcbb.run()

bb.print_stats()
