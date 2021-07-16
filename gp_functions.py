import autograd.numpy as np

def rbf_covariance(x, xp, kernel_params):
	output_scale = np.exp(kernel_params[0])
	lengthscales = np.exp(kernel_params[1:])
	diffs = np.expand_dims(x / lengthscales, 1)\
		  - np.expand_dims(xp / lengthscales, 0)
	return output_scale * np.exp(-0.5 * np.sum(diffs**2, axis=2))

def matrix_normal_logpdf(X, M, U, V):
	# import ipdb; ipdb.set_trace()
	n, p = X.shape
	assert M.shape == (n, p)
	assert U.shape == (n, n)
	assert V.shape == (p, p)

	V += 0.01 * np.eye(p)


	X_centered = X - M
	U_inv = np.linalg.solve(U, np.eye(n))
	V_inv = np.linalg.solve(V, np.eye(p))

	U_logdet = np.linalg.slogdet(U)[1]
	V_logdet = np.linalg.slogdet(V)[1]

	normalizer = -0.5 * n * p * np.log(2 * np.pi) - 0.5 * n * V_logdet - 0.5 * p * U_logdet
	# import ipdb; ipdb.set_trace()
	exponand  = -0.5 * np.trace(V_inv @ X_centered.T @ U_inv @ X_centered)

	
	return normalizer + exponand
