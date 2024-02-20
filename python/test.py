import test_add
import fsp_heuristics

print(test_add.adder(2,3))

inst=test_add.instance_taillard('ta20')

print(inst.get_job_number(20))
print(inst.get_machine_number(20))


bound=test_add.bound_fsp()
bound.init(inst)

s=test_add.subproblem(10)

print(s)

s=test_add.subproblem(20,[1,3,5,7,9,11,13,15,17,19,0,2,4,6,8,10,12,14,16,18])
print(s)

test_add.printvec(s.schedule)


print(bound.eval(s.schedule))


neh=test_add.fastNEH(inst)
#
# cost=0
neh.run(s)
print(s.schedule)
print(s)
