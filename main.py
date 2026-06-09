import numpy as np
import matplotlib.pyplot as plt

states = ['P', 'St', 'Sc']
state_index_map = {s: i for i, s in enumerate(states)}

reward_matrix = np.array([
    [0, 1, -1],
    [-1, 0, 1],
    [1, -1, 0]
])

repetitions = np.array([
    [1, 2, 1],
    [0, 1, 0],
    [0, 0, 1]
])

const_prob = np.array([0.5, 0.1, 0.4])

n = 100
state = 'P'
res_sum = 0
result_list = []

for i in range(n):
    arr = repetitions[state_index_map[state]]
    p_opp_pred = arr / arr.sum()
    expected_rewards = reward_matrix @ p_opp_pred

    pred_index = np.argmax(expected_rewards)
    pred = states[pred_index]

    opp_index = np.random.choice(len(states), p=const_prob)
    opp = states[opp_index]

    res = reward_matrix[pred_index, opp_index]
    res_sum += res
    result_list.append(res_sum)

    arr[state_index_map[opp]] += 1
    state = opp

    print(f"markov chain (my move): {pred}")
    print(f"opponent move: {opp}")
    print(f"result: {res}, repetitions:\n{repetitions}\n")

plt.plot(result_list)
plt.show()