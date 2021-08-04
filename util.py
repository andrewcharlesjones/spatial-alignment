import numpy as np
import pandas as pd
import numpy.random as npr

def polar_warp(X, r, theta):
	return np.array(
		[
			X[:, 0] + r * np.cos(theta),
			X[:, 1] + r * np.sin(theta)
		]).T


def get_st_coordinates(df):
    """
    Extracts spatial coordinates from ST data with index in 'AxB' type format.
	
    Return: pandas dataframe of coordinates
    """
    coor = []
    for spot in df.index:
        coordinates = spot.split('x')
        coordinates = [float(i) for i in coordinates]
        coor.append(coordinates)
    return np.array(coor)


def compute_distance(X1, X2):
    return np.mean(np.sqrt(np.sum((X1 - X2)**2, axis=1)))


def make_pinwheel(radial_std, tangential_std, num_classes, num_per_class, rate,
                  rs=npr.RandomState(0)):
    """Based on code by Ryan P. Adams."""
    rads = np.linspace(0, 2*np.pi, num_classes, endpoint=False)

    features = rs.randn(num_classes*num_per_class, 2) \
        * np.array([radial_std, tangential_std])
    features[:, 0] += 1
    labels = np.repeat(np.arange(num_classes), num_per_class)

    angles = rads[labels] + rate * np.exp(features[:,0])
    rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
    rotations = np.reshape(rotations.T, (-1, 2, 2))

    return np.einsum('ti,tij->tj', features, rotations)