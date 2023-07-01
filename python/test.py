import test_add

print(test_add.adder(2,3))

inst=test_add.instance_taillard('ta20')

print(inst.get_job_number(20))
print(inst.get_machine_number(20))


bound=test_add.bound_fsp()
bound.init(inst)
