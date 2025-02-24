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

    array[D] int<lower=-1,upper=1> vol_sign;     // Sign of volatility (1, 0 or -1)
    array[T] int recession_label; // Recession label (1 = recession, 0 = normal)


}


transformed data {

    int N_nonzero_vol_signs = 0;  // Number of non-zero volatilities
    for (d in 1:D){
        N_nonzero_vol_signs += vol_sign[d] != 0;
    }

    int N_shocks = sum(recession_label);  // Number of recession shocks
}

parameters {

    real<lower=-1,upper=1> phi_raw;    // Persistence of volatility  (raw version)
    array[P] matrix[D, D] A;           // VAR coefficients
    //vector[D] mu;                    // Intercepts (could be fixed at 0)
    vector[T] h;                       // Log volatility process
    real<lower=0> sigma_h;             // Volatility of log volatility process
    cov_matrix[D] Omega;               // Covariance matrix of residuals
    vector[D] mu_y;                    // is there some small deviation from 0 for y

    // Hidden Markov State (recession)
    vector[2] HMM_logit_transition_proba_intercept;  // Transition probabilities for hidden Markov state (normal -> recession and recession -> normal)
    vector[D] W_recession; // weights which multiply y to get an offset of logit probability of recession

    array[N_nonzero_vol_signs] real<lower=0> recession_shock_magnitude_raw; // magnitude of the shock to y in recession state
    array[N_shocks] vector<lower=0>[N_nonzero_vol_signs] recession_shock_realized_raw; 

}

transformed parameters {
    real phi = 2 * phi_raw - 1;  // Transform to (-1,1) 
    vector[T] volatility;
    array[N_shocks] vector[D] recession_shock_realized; 


    for (t in 1:T){
        volatility[t] = exp(h[t]);
    }

    {
    int count = 1;
    for (n in 1:N_shocks){

        count = 1;
        for (d in 1:D){
            recession_shock_realized[n, d] = recession_shock_realized_raw[n, count] * vol_sign[d];
            count += vol_sign[d] != 0;
        }
    }

    }
    

    
}

model {

    // declarations 

    // priors on HMM params
    HMM_logit_transition_proba_intercept[1] ~ normal(-5, 3);
    HMM_logit_transition_proba_intercept[2] ~ normal(-1, 2);
    W_recession ~ normal(0, 1/sqrt(D));
    recession_shock_magnitude_raw ~ exponential(1);  // Exponential prior on the magnitude of the shock


    // priors
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

        h[t] ~ normal(phi * h[t-1], sigma_h); // centered parametrization
    }

    // Prior for Omega (recall, y was normalized beforehand so they are unit variance)
    Omega ~ inv_wishart(D + 1, identity_matrix(D));


    // Hidden Markov State (recession) Likelihood
    recession_label[1] ~ bernoulli_logit(
        HMM_logit_transition_proba_intercept[1]
        + dot_product(W_recession, y[1, ])
        );  

    for (t in P+1:T) {
        recession_label[t] ~ bernoulli_logit(
            HMM_logit_transition_proba_intercept[recession_label[t - 1] + 1] 
            + dot_product(W_recession, y[t-1, ])
        );
    }

    for (n in 1:N_shocks){
        recession_shock_realized_raw[n] ~ exponential(recession_shock_magnitude_raw);
    }

    // Likelihood: VAR process with stochastic volatility
    int shock_ind = 1;
    for (t in (P + 1):T) {
        vector[D] y_temp = mu_y;

        for (p in 1:P) {
            y_temp += A[p] * to_vector(y[t - p, ]);  // Autoregressive part
        }
    
        if (recession_label[t] == 1) {
            y[t, ] ~ multi_normal(y_temp, exp(h[t]) * Omega);  // Stochastic volatility term
        } else {
            y[t, ] ~ multi_normal(y_temp + recession_shock_realized[shock_ind], exp(h[t]) * Omega);  // Stochastic volatility term
        }
        
    }
    

}


generated quantities {

    matrix[T, D] y_hat;                  // Posterior predictive check samples
    matrix[lookforward, D] y_forecast;   // Out-of-sample predictions
    vector[lookforward] h_forecast;      // Simulated volatility for out-of-sample forecast

    // declarations 
    array[T-P] int recession_state_hat;  // Hidden Markov state
    array[lookforward] int recession_state_pred;  // Hidden Markov state


    // Hidden Markov State (recession)
    recession_state_hat[1] = bernoulli_logit_rng(
        HMM_logit_transition_proba_intercept[1]
        + dot_product(W_recession, y[P, ])
        );  

    {
    array[N_nonzero_vol_signs] real recession_shock_realized_raw_t;  // Realized raw shock dummy
    vector[D] recession_shock_realized_t;  // Realized shock dummy

    // Posterior Predictive Checks (PPC)
    for (t in (P + 1):T) {
        vector[D] y_temp = mu_y;
        for (p in 1:P) {
            y_temp += A[p] * to_vector(y[t - p, ]);
        }
        if (recession_state_hat[t-P] == 1) {
            y_hat[t] = multi_normal_rng(y_temp, exp(h[t]) * Omega)';
        } else {

            recession_shock_realized_raw_t = exponential_rng(recession_shock_magnitude_raw);

            int count = 1;
            for (d in 1:D){
                recession_shock_realized_t[d] = recession_shock_realized_raw_t[count] * vol_sign[d];
                count += vol_sign[d] != 0;
            }

            y_hat[t, ] = multi_normal_rng(y_temp + recession_shock_realized_t, exp(h[t]) * Omega)';  // Stochastic volatility term
        }

        if (t < T) {
            recession_state_hat[t-P+1] = bernoulli_logit_rng(
                HMM_logit_transition_proba_intercept[recession_state_hat[t-P] + 1]
                + dot_product(W_recession, y[t-1, ])
            );
        }
        
    }
    }

    // Hidden Markov State (recession)
    recession_state_pred[1] = bernoulli_logit_rng(
        HMM_logit_transition_proba_intercept[recession_state_hat[T-P] + 1]
        + dot_product(W_recession, y[T, ])
        );

    // Out-of-Sample Forecasting
    {
        array[N_nonzero_vol_signs] real recession_shock_realized_raw_t;  // Realized raw shock dummy
        vector[D] recession_shock_realized_t;  // Realized shock dummy

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
            
            if (recession_state_pred[t] == 1) {
                y_forecast[t] = multi_normal_rng(y_temp, exp(h_forecast[t]) * Omega)';
            } else {
                recession_shock_realized_raw_t = exponential_rng(recession_shock_magnitude_raw);

                int count = 1;
                for (d in 1:D){
                    recession_shock_realized_t[d] = recession_shock_realized_raw_t[count] * vol_sign[d];
                    count += vol_sign[d] != 0;
                }
                y_forecast[t] = multi_normal_rng(y_temp + recession_shock_realized_t, exp(h[t]) * Omega)';  // Stochastic volatility term
            }

            if (t < lookforward) {
                recession_state_pred[t+1] = bernoulli_logit_rng(
                    HMM_logit_transition_proba_intercept[recession_state_pred[t] + 1] 
                    + dot_product(W_recession, y_forecast[t, ])
                    );
            }
            
            
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