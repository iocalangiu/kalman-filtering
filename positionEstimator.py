import numpy as np

def positionEstimator(trial, ModelParameters):
    dt = 20


    if trial['spikes'].shape[1] == 320:
        angleDECODED = ModelParameters[0]['direction']
        starting = trial['startHandPos']
        x = np.array([starting[0], starting[1], 0, 0, 0, 0])
        ModelParameters[0]['errorCovariance'] = np.eye(6)
        neurons = np.arange(1, 99)
        ModelParameters[0]['kalmanGain'] = np.zeros((6, len(neurons)))
    elif trial['spikes'].shape[1] != 320:
        angleDECODED = ModelParameters[0]['direction']
        A = ModelParameters[angleDECODED]['A']
        H = ModelParameters[angleDECODED]['H']
        W = ModelParameters[angleDECODED]['W']
        Q = ModelParameters[angleDECODED]['Q']
        Pcov_error = ModelParameters[0]['errorCovariance']
        K = ModelParameters[0]['kalmanGain']
        velocity = ModelParameters[0]['velocity']
        acceleration = ModelParameters[0]['acceleration']
        decoded = trial['decodedHandPos'][-1]
        x = np.array([decoded[0], decoded[1], velocity[0], velocity[1], acceleration[0], acceleration[1]])

        time = trial['spikes'].shape[1] - dt
        neurons = np.arange(0, 98)

        firingRateTest = np.zeros(len(neurons))
        for neuron in range(len(neurons)):
            n = np.sum(trial['spikes'][neurons[neuron], time:time+dt])
            firingRateTest[neuron] = n

        x_Prior = np.dot(A, x)
        Pcov_error = np.dot(np.dot(A, Pcov_error), np.transpose(A)) + W

        K = np.dot(np.dot(Pcov_error, np.transpose(H)), np.linalg.pinv(np.dot(np.dot(H, Pcov_error), np.transpose(H)) + Q))
        
        x = x_Prior + np.dot(K, firingRateTest - np.dot(H, x_Prior))
        Pcov_error = np.dot(np.eye(A.shape[0]) - np.dot(K, H), Pcov_error)

        ModelParameters[0]['errorCovariance'] = Pcov_error
        ModelParameters[0]['kalmanGain'] = K
        ModelParameters[0]['direction'] = angleDECODED
        ModelParameters[0]['velocity'] = [x[2], x[3]]
        ModelParameters[0]['acceleration'] = [x[4], x[5]]

    x1, y1 = x[0], x[1]
    return x1, y1, ModelParameters
