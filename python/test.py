import test_add

print(test_add.adder(2,3))

inst=test_add.instance_taillard('ta20')

print(inst.get_job_number(20))
print(inst.get_machine_number(20))


bound=test_add.bound_fsp()
bound.init(inst)

s=test_add.subproblem(10)

print(s)

s=test_add.subproblem(10,[1,3,5,7,9,0,2,4,6,8])
print(s)

print(bound.eval(s.schedule))
