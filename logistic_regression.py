import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use Agg backend to avoid GUI issues
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import cdist
import os

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

def generate_ellipsoid_clusters(distance, n_samples=100, cluster_std=0.5):
    np.random.seed(0)
    covariance_matrix = np.array([[cluster_std, cluster_std * 0.8], 
                                 [cluster_std * 0.8, cluster_std]])
    
    # Generate the first cluster (class 0)
    X1 = np.random.multivariate_normal(mean=[1, 1], cov=covariance_matrix, size=n_samples)
    y1 = np.zeros(n_samples)

    # Generate the second cluster (class 1)
    X2 = np.random.multivariate_normal(mean=[1, 1], cov=covariance_matrix, size=n_samples)
    
    # Shift the second cluster along y = -x direction
    # This means moving positive in x and negative in y
    X2 += np.array([distance, -distance])  # Changed from [distance, distance] to [distance, -distance]
    y2 = np.ones(n_samples)

    # Combine the clusters into one dataset
    X = np.vstack((X1, X2))
    y = np.hstack((y1, y2))
    return X, y

def calculate_logistic_loss(model, X, y):
    # Calculate probabilities
    probs = model.predict_proba(X)
    # Calculate log loss (cross-entropy)
    epsilon = 1e-15  # Small constant to avoid log(0)
    probs = np.clip(probs, epsilon, 1 - epsilon)
    loss = -np.mean(y * np.log(probs[:, 1]) + (1 - y) * np.log(probs[:, 0]))
    return loss

def fit_logistic_regression(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    beta0 = model.intercept_[0]
    beta1, beta2 = model.coef_[0]
    return model, beta0, beta1, beta2

def do_experiments(start, end, step_num):
    # Clear any existing plots at the start only
    plt.clf()
    plt.close('all')
    
    # Set up experiment parameters
    shift_distances = np.linspace(start, end, step_num)
    beta0_list, beta1_list, beta2_list = [], [], []
    slope_list, intercept_list, loss_list, margin_widths = [], [], [], []

    n_samples = step_num
    n_cols = 2
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    plt.figure(figsize=(20, n_rows * 10))

    # Run experiments for each shift distance
    for i, distance in enumerate(shift_distances, 1):
        # Generate and fit data
        X, y = generate_ellipsoid_clusters(distance=distance)
        model, beta0, beta1, beta2 = fit_logistic_regression(X, y)
        
        # Record parameters
        beta0_list.append(beta0)
        beta1_list.append(beta1)
        beta2_list.append(beta2)
        
        # Calculate slope and intercept
        slope = -beta1 / beta2
        intercept = -beta0 / beta2
        slope_list.append(slope)
        intercept_list.append(intercept)
        
        # Calculate loss
        loss = calculate_logistic_loss(model, X, y)
        loss_list.append(loss)

        # Plot the dataset
        plt.subplot(n_rows, n_cols, i)
        plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='blue', label='Class 0', alpha=0.6)
        plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='red', label='Class 1', alpha=0.6)

        # Calculate margin width between 70% confidence contours for each class
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)

        # Plot decision boundary
        plt.plot([x_min, x_max], [slope * x_min + intercept, slope * x_max + intercept], 
                'k-', label='Decision Boundary')

        # Plot confidence contours
        contour_levels = [0.7, 0.8, 0.9]
        alphas = [0.05, 0.1, 0.15]
        for level, alpha in zip(contour_levels, alphas):
            class_1_contour = plt.contourf(xx, yy, Z, levels=[level, 1.0], colors=['red'], alpha=alpha)
            class_0_contour = plt.contourf(xx, yy, Z, levels=[0.0, 1 - level], colors=['blue'], alpha=alpha)
            if level == 0.7:
                distances = cdist(class_1_contour.collections[0].get_paths()[0].vertices,
                                class_0_contour.collections[0].get_paths()[0].vertices,
                                metric='euclidean')
                min_distance = np.min(distances)
                margin_widths.append(min_distance)

        plt.title(f"Shift Distance = {distance:.2f}", fontsize=24)
        plt.xlabel("x1")
        plt.ylabel("x2")
        
        # Get the current axes
        ax = plt.gca()

        # Prepare the text content
        equation_text = f"{beta0:.2f} + {beta1:.2f} * x1 + {beta2:.2f} * x2 = 0\nx2 = {slope:.2f} * x1 + {intercept:.2f}"
        margin_text = f"Margin Width: {min_distance:.2f}"

        # Position the text boxes using axes coordinates
        ax.text(0.05, 0.95, equation_text, transform=ax.transAxes, fontsize=24, verticalalignment='top',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

        ax.text(0.05, 0.80, margin_text, transform=ax.transAxes, fontsize=24, verticalalignment='top',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

        if i == 1:
            plt.legend(loc='lower right', fontsize=20)

    plt.tight_layout()
    plt.savefig(f"{result_dir}/dataset.png")
    plt.close()

    # Create parameter plots
    plt.figure(figsize=(18, 15))

    # Plot beta0
    plt.subplot(3, 3, 1)
    plt.plot(shift_distances, beta0_list, 'b-', marker='o')
    plt.title("Shift Distance vs Beta0")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta0")

    # Plot beta1
    plt.subplot(3, 3, 2)
    plt.plot(shift_distances, beta1_list, 'g-', marker='o')
    plt.title("Shift Distance vs Beta1")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta1")

    # Plot beta2
    plt.subplot(3, 3, 3)
    plt.plot(shift_distances, beta2_list, 'r-', marker='o')
    plt.title("Shift Distance vs Beta2")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta2")

    # Plot slope
    plt.subplot(3, 3, 4)
    plt.plot(shift_distances, slope_list, 'k-', marker='o')
    plt.title("Shift Distance vs Slope")
    plt.xlabel("Shift Distance")
    plt.ylabel("Slope")

    # Plot intercept
    plt.subplot(3, 3, 5)
    plt.plot(shift_distances, intercept_list, 'm-', marker='o')
    plt.title("Shift Distance vs Intercept")
    plt.xlabel("Shift Distance")
    plt.ylabel("Intercept")

    # Plot loss
    plt.subplot(3, 3, 6)
    plt.plot(shift_distances, loss_list, 'y-', marker='o')
    plt.title("Shift Distance vs Loss")
    plt.xlabel("Shift Distance")
    plt.ylabel("Loss")

    # Plot margin width
    plt.subplot(3, 3, 7)
    plt.plot(shift_distances, margin_widths, 'c-', marker='o')
    plt.title("Shift Distance vs Margin Width")
    plt.xlabel("Shift Distance")
    plt.ylabel("Margin Width")

    plt.tight_layout()
    plt.savefig(f"{result_dir}/parameters_vs_shift_distance.png")
    plt.close()

if __name__ == "__main__":
    start = 0.25
    end = 2.0
    step_num = 8
    do_experiments(start, end, step_num)