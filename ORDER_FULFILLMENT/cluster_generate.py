import os

num_items = 20
n_0 = 5
p_stock = 0.75
num_instances = 30

for instance in range(1, num_instances + 1):
    slurm_dir = f'slurm_instance_{instance}'

    if not os.path.exists(slurm_dir):
            os.makedirs(slurm_dir)

    # Names of the files to be read and copied
    source_files = ['order_fulfillment_environment_notidentical_arrival_probs.py',
                    'LP_fulfillment_notidentical_arrival_probs.py',
                    'cluster_fulfillment_policy_notidentical_arrival_probs.py',
                    'magician_problem.py',
                    'LP_fulfillment_item_facility_based.py',
                    'fulfillment_policy_Will_Ma.py']

    # Copying the content of each file into the 'slurm_instance_{instance}' directory
    for file_name in source_files:
        with open(os.path.join('ORDER_FULFILLMENT', file_name), 'r') as source_file:
            content = source_file.read()

        with open(os.path.join(slurm_dir, file_name), 'w') as destination_file:
            destination_file.write(content)

    script = []

    with open('ORDER_FULFILLMENT/cluster_simulation.py', 'r') as f:
        line = f.readline()
        while line:
            script.append(line)
            line = f.readline()

    T_values = [10**3]
    # alpha_values = [0.1, 0.5, 0.75, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 1]
    # alpha_values = [0.5, 0.9, 1]
    alpha_values = [0, 0.5, 1]
    # n_max_values = [2, 5, 10]
    n_max_values = [2, 5]

    script = script[:-1]
    for n_max in n_max_values:
        for T in T_values:
            for alpha in alpha_values:
                new_script = script.copy()
                new_script.append(r'evaluate_policies(n_max=%s, T=%s, alpha=%s, instance=%s, num_order_sequences=num_order_sequences, num_items=num_items, n_0=n_0, p_stock=p_stock, conservative_prob_sequence=conservative_prob_sequence)' % (n_max, T, alpha, instance))
                
                with open('%s/cluster_simulation_n_max=%s_T_%s_alpha_%s.py' % (slurm_dir, n_max, T, alpha), 'w') as f:
                    for line in new_script:
                        f.write(line.rstrip('\n'))
                        f.write('\n')

                with open('%s/cluster_simulation_n_max=%s_T_%s_alpha_%s.sh' % (slurm_dir, n_max, T, alpha) , "w") as output_file:
                        output_file.write(r'#!/bin/bash' + '\n')
                        output_file.write(r'#SBATCH -e ./slurm_out/cluster_simulation_n_max=%s_T_%s_alpha_%s.err' % (n_max, T, alpha) + '\n')
                        output_file.write(r'#SBATCH -o ./slurm_out/cluster_simulation_n_max=%s_T_%s_alpha_%s.out' % (n_max, T, alpha) + '\n')
                        output_file.write(r'#SBATCH --partition=econ,scavenger' + '\n')
                        output_file.write(r'#SBATCH --mail-type=begin #send email when job begins' + '\n')
                        output_file.write(r'#SBATCH --mail-type=end   #send email when job ends' + '\n')
                        output_file.write(r'#SBATCH --mail-user=aa554@duke.edu' + '\n')
                        output_file.write(r'#SBATCH --mem-per-cpu=4G  # adjust as needed' + '\n')
                        output_file.write(r'#SBATCH -c 4 # CPU cores, adjust as needed' + '\n')
                        output_file.write(r'module purge' + '\n')
                        output_file.write(r'module load /opt/apps/modules-bak/Python/3.8.1' + '\n')
                        # output_file.write(r'module load Python/3.8.1' + '\n')

                        new_line = r'python cluster_simulation_n_max=%s_T_%s_alpha_%s.py' % (n_max, T, alpha)
                        output_file.write(new_line)