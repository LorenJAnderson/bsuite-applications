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
                             'ids': [0, 1, 2, 3, 4],
                             'time_steps': [1_000_000]*5,
                             'fixed': False
                             },
            'mountain_car_scale': {'env_name': 'mountain_car_scale',
                                   'ids': [0, 4, 8, 12, 16],
                                   'time_steps': [1_000_000]*5,
                                   'fixed': False
                                   },
            'deep_sea': {'env_name': 'deep_sea',
                         'ids': [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                         'time_steps': [100_000, 140_000, 180_000, 220_000,
                                        260_000, 300_000, 340_000, 380_000,
                                        420_000, 460_000, 500_000],
                         'fixed': True
                         },
            'deep_sea_stochastic': {'env_name': 'deep_sea_stochastic',
                             'ids': [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                             'time_steps': [100_000, 140_000, 180_000, 220_000,
                                        260_000, 300_000, 340_000, 380_000,
                                        420_000, 460_000, 500_000],
                             'fixed': True
                             },
            'umbrella_length': {'env_name': 'umbrella_length',
                                'ids': [0, 1, 3, 5, 8, 11, 16, 22],
                                'time_steps': [10_000, 20_000, 40_000, 60_000,
                                             90_000, 150_000, 410_000, 1_010_000],
                                'fixed': True
                                },
            'umbrella_distract': {'env_name': 'umbrella_distract',
                                  'ids': [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
                                  'time_steps': [200_000] * 12,
                                  'fixed': True
                                  },
            'discounting_chain': {'env_name': 'discounting_chain',
                                  'ids': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                  'time_steps': [100_000] * 10,
                                  'fixed': True
                                  },
            'memory_len': {'env_name': 'memory_len',
                              'ids': [0, 1, 3, 5, 8, 11, 16, 22],
                              'time_steps': [20_000, 30_000, 50_000, 70_000,
                                             100_000, 160_000, 420_000, 1_020_000],
                              'fixed': True
                              },
            'memory_size': {'env_name': 'memory_size',
                            'ids': [0, 2, 4, 6, 8, 10, 12, 14, 16],
                            'time_steps': [30_000] * 9,
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
