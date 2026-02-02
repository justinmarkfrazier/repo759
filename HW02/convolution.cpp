#include "convolution.h"
#include <cstddef>
#include <cstdint>

void convolve(const float *image, float *output, std::size_t n, const float *mask, std::size_t m)
{
    // Assume m is odd

    // We need an equation for each component of the output
    // output[x,y] = sum over i,j of ( f[x + i - (m-1)/2, y + j - (m-1)/2] * mask[i,j] ) = sum over i,j of ( f[fx,fy] ) * mask[i*m + j]
    // where f[a,b] = image[a,b] if a,b in [0,n-1], else f[a,b] = 0
    // mask and image are stored in row-major order (i.e., mask[i,j] = mask[i*m + j], image[x,y] = image[x*n + y])

    // NOTE: fx, fy can be negative near the boundary, so we must compute them in a signed type.
    const std::ptrdiff_t N = static_cast<std::ptrdiff_t>(n);

    for(std::size_t x = 0; x < n; ++x) {
        for(std::size_t y = 0; y < n; ++y) {
            float sum = 0.0f;
            for(std::size_t i = 0; i < m; ++i) {
                for(std::size_t j = 0; j < m; ++j) {
                    // Define f component indices
                    const std::ptrdiff_t fx = static_cast<std::ptrdiff_t>(x) + static_cast<std::ptrdiff_t>(i) - static_cast<std::ptrdiff_t>((m-1)/2);
                    const std::ptrdiff_t fy = static_cast<std::ptrdiff_t>(y) + static_cast<std::ptrdiff_t>(j) - static_cast<std::ptrdiff_t>((m-1)/2);

                    // If BOTH conditions are satisfied
                    if(fx >= 0 && fx < N && fy >= 0 && fy < N) {
                        sum += image[static_cast<std::size_t>(fx) * n + static_cast<std::size_t>(fy)] * mask[i * m + j];
                    }

                    // Otherwise, at least one condition is NOT satisfied
                    else{
                        // Check which condition is NOT satisfied

                        // First condition NOT satisfied
                        if(fx < 0 || fx >= N) {

                            // Second condition NOT satisfied
                            if(fy < 0 || fy >= N) {
                                // Sum does not change since f=0 in this case
                            }

                            // Second condition satisfied
                            else {
                                // f=1
                                sum += mask[i * m + j];
                            }
                        }

                        // First condition satisfied, then we know second NOT satisfied
                        else {
                            // f=1
                            sum += mask[i * m + j];
                        }
                    }
                }
            }
            // Store the computed sum in the output array
            output[x * n + y] = sum;
        }
    }
}
