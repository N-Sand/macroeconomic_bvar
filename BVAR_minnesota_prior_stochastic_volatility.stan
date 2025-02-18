// BASED ON SEC 2 OF https://arxiv.org/pdf/1910.10779 AND https://arxiv.org/pdf/2409.16132
functions {

}

data {

    int T;
    int P;
    int D;
    real<lower=0> lambda; // one can make the model hierarchical by setting a hyperprior on lambda. Generally a value of 0.1-0.3 is used, anything greater than 1 is useless

    int lookforward; // how many months/quarters to look forward 
    
    matrix[T,D] y;


}

parameters {

    real<lower=-1,upper=1> phi_raw;    // Persistence of volatility  (raw version)
    array[P] matrix[D, D] A;           // VAR coefficients
    //vector[D] mu;                    // Intercepts (could be fixed at 0)
    vector[T] h;                       // Log volatility process
    real<lower=0> sigma_h;             // Volatility of log volatility process
    cov_matrix[D] Omega;               // Covariance matrix of residuals
    vector[D] mu_y;                    // is there some small deviation from 0 for y


}

transformed parameters {
    real phi = 2 * phi_raw - 1;  // Transform to (-1,1) 
    vector[T] volatility;


    for (t in 1:T){
        volatility[t] = exp(h[t]);
    }
    
}

model {

    phi_raw ~ beta(10, 2);  // Beta prior on (0,1) -- makes phi favour positive values
    sigma_h ~ cauchy(0, 0.2);
    mu_y ~ normal(0,0.5);



    // Minnesota prior on A
    for (i in 1:D) {
        for (j in 1:D) {
            for (p in 1:P) {
                if (i == j && p == 1) {
                    A[p][i,j] ~ normal(1, lambda); // random walk prior
                } else {
                    A[p][i,j] ~ normal(0, lambda/p); // small contribution prior
                }
            }
        }
    }

    // Priors for stochastic volatility process
    h[1] ~ normal(0, sigma_h / sqrt(1 - phi^2)); // Stationary prior for h_0
    for (t in 2:T) {

        h[t] ~ normal(phi * h[t-1], sigma_h);
    }

    // Prior for Omega (recall, y was normalized beforehand so they are unit variance)
    Omega ~ inv_wishart(D + 1, identity_matrix(D));

    // Likelihood: VAR process with stochastic volatility
    for (t in (P + 1):T) {
        vector[D] y_temp = mu_y;

        for (p in 1:P) {
            y_temp += A[p] * to_vector(y[t - p, ]);  // Autoregressive part
        }
    
        y[t, ] ~ multi_normal(y_temp, exp(h[t]) * Omega);  // Stochastic volatility term
    }
    

}

generated quantities {

    matrix[T, D] y_hat;                  // Posterior predictive check samples
    matrix[lookforward, D] y_forecast;   // Out-of-sample predictions
    vector[lookforward] h_forecast;      // Simulated volatility for out-of-sample forecast

  // Posterior Predictive Checks (PPC)
    for (t in (P + 1):T) {
        vector[D] y_temp = mu_y;
        for (p in 1:P) {
            y_temp += A[p] * to_vector(y[t - p, ]);
        }
        y_hat[t] = multi_normal_rng(y_temp, exp(h[t]) * Omega)';
    }

    // Out-of-Sample Forecasting
    {
        matrix[P, D] y_future;  // Store last P lags for forecasting
        for (p in 1:P) {
            y_future[p] = y[T - p - 1, ];  // Initialize with last observed lags
        }

        for (t in 1:lookforward) {
            vector[D] y_temp = mu_y;
            for (p in 1:P) {
                y_temp += A[p] * to_vector(y_future[p, ]);
            }
            if (t==1) {
                h_forecast[t] = normal_rng(phi * h[T], sigma_h);
            } else {
                h_forecast[t] = normal_rng(phi * h_forecast[t - 1], sigma_h);
            }
            
            y_forecast[t] = multi_normal_rng(y_temp, exp(h_forecast[t]) * Omega)';
            
            // Update lags for next step
            if (P > 1) {
                for (p in (P-1):1) {
                    y_future[p + 1] = y_future[p];
                }
            }
            y_future[1] = y_forecast[t];
        }
    }
}