def sweep_dict() -> dict:
    return {'bandit': {'env_name': 'bandit',
                       'ids': list(range(20)),
                       'time_steps': [10_000]*20,
                       'fixed': True
                       },
            'bandit_noise': {'env_name': 'bandit_noise',
                             'ids': list(range(20)),
                             'time_steps': [10_000]*20,
                             'fixed': True
                             },
            'bandit_scale': {'env_name': 'bandit_scale',
                             'ids': list(range(20)),
                             'time_steps': [10_000]*20,
                             'fixed': True
                             },
            'mountain_car': {'env_name': 'mountain_car',
                             'ids': [0, 1],
                             'time_steps': [1_000_000]*2,
                             'fixed': False
                             },
            'mountain_car_noise': {'env_name': 'mountain_car_noise',
                                   'ids': [0, 1],
                                   'time_steps': [1_000_000]*2,
                                   'fixed': False
                                   },
            'mountain_car_scale': {'env_name': 'mountain_car_scale',
                                   'ids': [0, 1],
                                   'time_steps': [1_000_000]*2,
                                   'fixed': False
                                   },
            'deep_sea': {'env_name': 'deep_sea',
                         'ids': [0, 5, 10],
                         'time_steps': [100_000, 200_000, 300_000],
                         'fixed': True
                         },
            'deep_sea_stochastic': {'env_name': 'deep_sea_stochastic',
                             'ids': [0, 5, 10],
                             'time_steps': [100_000, 200_000, 300_000],
                             'fixed': True
                             },
            'umbrella_length': {'env_name': 'umbrella_length',
                                'ids': [0, 5],
                                'time_steps': [10_000, 60_000],
                                'fixed': True
                                },
            'umbrella_distract': {'env_name': 'umbrella_distract',
                                  'ids': [0, 5],
                                  'time_steps': [200_000] * 2,
                                  'fixed': True
                                  },
            'discounting_chain': {'env_name': 'discounting_chain',
                                  'ids': [0, 4],
                                  'time_steps': [100_000] * 2,
                                  'fixed': True
                                  },
            'memory_len': {'env_name': 'memory_len',
                              'ids': [0, 5],
                              'time_steps': [20_000, 70_000],
                              'fixed': True
                              },
            'memory_size': {'env_name': 'memory_size',
                            'ids': [0, 5, 11, 16],
                            'time_steps': [30_000] * 4,
                            'fixed': True
                            },
            }


def sweep_list() -> list:
    master_dict = sweep_dict()
    master_list = []
    for key in master_dict:
        experiment = master_dict[key]
        for idx, id_number in enumerate(experiment['ids']):
            env_name = experiment['env_name'] + '/'+ str(experiment['ids'][idx])
            time_steps = experiment['time_steps'][idx]
            fixed = experiment['fixed']
            csv_name = 'bsuite_id_-_' + experiment['env_name'] + '-' + str(experiment['ids'][idx]) + '.csv'
            master_list.append([env_name, time_steps, fixed, csv_name])
    return master_list


if __name__ == "__main__":
    print(sweep_list())
