import numpy as np

def positionEstimatorTraining(trial):
    neurons = np.arange(0, 98)
    angleDECODED = 0
    P_cov = np.eye(6)
    K = np.zeros((6, len(neurons)))
    vx = 0
    vy = 0
    ax = 0
    ay = 0
    dt = 20

    ModelParameters = [{
        'A': None,
        'H': None,
        'W': None,
        'Q': None,
        'errorCovariance': P_cov,
        'kalmanGain': K,
        'direction': angleDECODED,
        'lambda': None,
        'velocity': [vx, vy],
        'acceleration': [ax, ay],
        'bayesian': None,
        'lambda2': None
    } for _ in range(8)]

    for angle in range(8):
        A_mean = np.zeros((6, 6))
        H_mean = np.zeros((len(neurons), 6))
        W_mean = np.zeros((6, 6))
        Q_mean = np.zeros((len(neurons), len(neurons)))

        for trl in range(len(trial)):
            positionx = trial[trl][angle]['handPos'][0][::dt]          
            positiony = trial[trl][angle]['handPos'][1][::dt]
            
            velocityx = np.diff(positionx) / dt
            velocityy = np.diff(positiony) / dt
            # Prepend '0' to velocityx and velocityy
            velocityx = np.insert(velocityx, 0, 0)
            velocityy = np.insert(velocityy, 0, 0)

            accelerationx = np.diff(velocityx) / dt
            accelerationy = np.diff(velocityy) / dt
            # Prepend '0' to accelerationx and accelerationy
            accelerationx = np.insert(accelerationx, 0, 0)
            accelerationy = np.insert(accelerationy, 0, 0)


            X = np.array([
                positionx[1:],
                positiony[1:],
                velocityx[1:],
                velocityy[1:],
                accelerationx[1:],
                accelerationy[1:]
            ])

            X2 = X[:, 1:]
            X1 = X[:, :-1]
            A = np.dot(X2, X1.T) @ np.linalg.pinv(X1 @ X1.T)

            #Z = np.zeros((len(neurons), len(X.T)))

            Z = []
            for neuron in range(len(neurons)):
                firingRate = np.array([sum(trial[trl][angle]['spikes'][neurons[neuron], k:k+dt]) for k in range(0, len(trial[trl][angle]['spikes'][neurons[neuron]])-dt, dt)])
                Z.append(firingRate)

            H = np.dot(Z, X.T) @ np.linalg.pinv(X @ X.T)

            W = np.dot((X2 - A @ X1), (X2 - A @ X1).T) / len(accelerationx)
            Q = np.dot((Z - H @ X), (Z - H @ X).T) / len(accelerationx)

            A_mean += A
            H_mean += H
            W_mean += W
            Q_mean += Q

        A_mean /= len(trial)
        H_mean /= len(trial)
        W_mean /= len(trial)
        Q_mean /= len(trial)

        ModelParameters[angle]['A'] = A_mean
        ModelParameters[angle]['H'] = H_mean
        ModelParameters[angle]['W'] = W_mean
        ModelParameters[angle]['Q'] = Q_mean

    lambda_values = np.zeros((len(neurons), 8))
    lambda2_values = np.zeros((len(neurons), 8))

    for angle in range(8):
        for neuron in range(len(neurons)):
            lambda_values[neuron, angle] = np.mean([sum(trial[trl][angle]['spikes'][neurons[neuron], 50:320]) for trl in range(len(trial))])
            lambda2_values[neuron, angle] = np.mean([sum(trial[trl][angle]['spikes'][neurons[neuron], 50:340]) for trl in range(len(trial))])

    ModelParameters[0]['lambda'] = lambda_values
    ModelParameters[0]['lambda2'] = lambda2_values

    return ModelParameters