from collections import namedtuple

# task specific params
TaskTSP = namedtuple('TaskTSP', ['task_name', 
						'input_dim', 
						'n_nodes',
						'decode_len'])
TaskVRP = namedtuple('TaskVRP', ['task_name', 
						'input_dim',
						'n_nodes' ,
						'n_cust',
						'decode_len',
						'capacity',
						'demand_max'])


task_lst = {}

# VRP100
manu = TaskVRP(task_name = 'vrp',
			  num_of_orders=420,
              num_unique_orders=132,
			  num_production_lines=3,
			  labour_max=2000)
task_lst['manu'] = manu