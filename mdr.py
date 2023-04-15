import numpy as np
import matplotlib.pyplot as plt

# Definindo as variáveis do problema
num_states = 3
num_actions = 2
P = np.zeros((num_states, num_actions, num_states))
R = np.zeros((num_states, num_actions, num_states))

# Preenchendo as matrizes de transição e recompensa

#   _____  
#  |  __ \ 
#  | |__) |
#  |  ___/ 
#  | |     
#  |_|     

# Preenchendo P -> Matriz de transição
# P[s, a, s'] = probabilidade de ir do estado s para o estado s' ao executar a ação a

# =================================================================================================
# estado 0 (frio) -> ação 0 (devagar) -> vai para 0 (frio) com prob 1.0 ou vai para 1 (quente) com prob 0.0
P[0 , 0 , 0] = 1.0

# estado 0 (frio) -> ação 1 (rapido) -> vai para 0 (frio) com prob 0.5 ou vai para 1 (quente) com prob 0.5
P[0 , 1 , 0] = 0.5
P[0 , 1 , 1] = 0.5

# estado 1 (quente) -> ação 0 (devagar) -> vai para 0 (frio) com prob 0.5 ou vai para 1 (quente) com prob 0.5
P[1 , 0 , 0] = 0.5
P[1 , 0 , 1] = 0.5
# estado 1 (quente) -> ação 1 (rapido) -> vai para 2 (fogo) com prob 1.0
P[1 , 1 , 2] = 1.0

# estado 2 (fogo) -> ação 0 (devagar) -> vai para 2 (fogo) com prob 1.0
P[2 , 0 , 2] = 1.0

# estado 2 (fogo) -> ação 1 (rapido) -> vai para 2 (fogo) com prob 1.0
P[2 , 1 , 2] = 1.0
# =================================================================================================



#   _____
#  |  __ \ 
#  | |__) |
#  |  _  / 
#  | | \ \ 
#  |_|  \_\

# Preenchendo R -> Matriz de recompensa
# R[s, a, s'] = recompensa ao ir do estado s para o estado s' ao executar a ação a (deve ser um valor numérico)

# =================================================================================================

# estado 0 (frio) -> ação 0 (devagar) -> recompensa 1.0 ao ir para 0 (frio) ou 0.0 ao ir para 1 (quente)
R[0, 0, 0] = 1.0

# estado 0 (frio) -> ação 1 (rapido) -> recompensa 2.0 ao ir para 0 (frio) ou 2.0 ao ir para 1 (quente)
R[0, 1, 0] = 2.0
R[0, 1, 1] = 2.0

# estado 1 (quente) -> ação 0 (devagar) -> recompensa 1.0 ao ir para 0 (frio) ou 1.0 ao ir para 1 (quente)
R[1, 0, 0] = 1.0
R[1, 0, 1] = 1.0
# estado 1 (quente) -> ação 1 (rapido) -> recompensa -1.0 ao ir para 2 (fogo)
R[1, 1, 2] = -1.0

# estado 2 (fogo) -> ação 0 (devagar) -> recompensa -1.0 ao ir para 2 (fogo)
R[2, 0, 2] = -1.0

# estado 2 (fogo) -> ação 1 (rapido) -> recompensa -1.0 ao ir para 2 (fogo)
R[2, 1, 2] = -1.0

# =================================================================================================

# Definindo as variáveis para a iteração
gamma = 0.9
max_iterations = 100000
epsilon = 0.000000001
V = np.zeros(num_states)

# Algoritmo de Value Iteration
# Calculando o valor ótimo para cada estado do problema de MDP
# V(s) = max_a sum_s' P(s'|s,a) * (R(s'|s,a) + gamma * V(s'))
for i in range(max_iterations):
    delta = 0
    for s in range(num_states):
        v = V[s]
        q_values = np.zeros(num_actions)
        for a in range(num_actions):
            q_value = 0
            for s_prime in range(num_states):
                q_value += P[s, a, s_prime] * (R[s, a, s_prime] + gamma * V[s_prime])
            q_values[a] = q_value
        V[s] = np.max(q_values)
        delta = max(delta, np.abs(v - V[s]))
    if delta < epsilon:
        break

# Encontrando a política ótima
# pi(s) = argmax_a sum_s' P(s'|s,a) * (R(s'|s,a) + gamma * V(s'))
policy = np.zeros(num_states)
for s in range(num_states):
    q_values = np.zeros(num_actions)
    for a in range(num_actions):
        q_value = 0
        for s_prime in range(num_states):
            q_value += P[s, a, s_prime] * (R[s, a, s_prime] + gamma * V[s_prime])
        q_values[a] = q_value
    policy[s] = np.argmax(q_values)

# Encontrando o caminho de tamanho n iniciando no estado frio
def find_path(n):
    path = [0]
    state = 0
    for i in range(n):
        action = int(policy[state])
        next_state = np.random.choice(num_states, p=P[state, action])
        path.append(next_state)
        state = next_state
        if state == 2:
            break
    return path

print('Valor ótimo de cada estado (valor esperado de recompensa ao iniciar no estado e seguir a política ótima):')
print('\tEstado frio: ', V[0])
print('\tEstado quente: ', V[1])
print('\tEstado fogo: ', V[2])


print('Política ótima de cada estado (decisão a se tomar em cada estado): ')
print('\tEstado frio: ', policy[0] , ' (0 = devagar, 1 = rapido)')
print('\tEstado quente: ', policy[1] , ' (0 = devagar, 1 = rapido)')
print('\tEstado fogo: ', policy[2] , ' (0 = devagar, 1 = rapido)')

# Encontrando um caminho de tamanho 100 iniciando no estado frio

print(find_path(100))
