/* Spline.h
   Interpolate 1D, 2D, 3D with basis splines
   */

#include <iostream>
#include <constants.h>
#include <math.h>


inline int modulo(int a, int b) {
    int r = a % b;
    return (r < 0) ? r+b : r;
}


template<int DIM>
class spline{
public:
    spline(const Float xmin[DIM], const Float xmax[DIM],  const int N[DIM]){
        uint datasize = 1;
        for(int i=0; i < DIM; i++){
            this->xmin[i] = xmin[i];
            this->xmax[i] = xmax[i];
            this->N[i] = N[i];
            xres[i] = (N[i]-1)/(xmax[i] - xmin[i]);
            xres2[i] = xres[i]*xres[i];
            datasize= datasize*N[i];
        }
        coeff = new Float[datasize];
        built = false;
        z1 = -2 + std::sqrt(3);
        
    }
    ~spline(){
        delete[] coeff;
    }

    void build(const Float data[]){
        if(DIM == 1){
            build1d(data, 0, 1, N[0], coeff);
            return;
        }
        if(DIM == 2){
            build2d(data);
            return;
        }
        if(DIM == 3){
            build3d(data);
            return;
        }
	    std::cerr << "Error: Current implementation of spline interpolation works only upto 3 dimensions \n";
        exit (EXIT_FAILURE);
    }

    template<int DX>
    inline Float value(const Float x[]) const{
        if(DIM != 1){
	        std::cerr << "Error: 1D-Value called for " << DIM << "dimensions \n";
            exit (EXIT_FAILURE);
        }

        Float y[DIM]; // make a duplicate
        y[0] = x[0];
        return value1d<DX>(y);
    }

    template<int DX, int DY>
    inline Float value(const Float x[]) const{
        if(DIM != 2){
	        std::cerr << "Error: 2D-Value called for " << DIM << "dimensions \n";
            exit (EXIT_FAILURE);
        }

        Float y[DIM]; // make a duplicate
        y[0] = x[0]; y[1] = x[1];
        return value2d<DX, DY>(y);
    }

    template<int DX, int DY, int DZ>
    inline Float value(const Float x[]) const{
        if(DIM != 3){
	        std::cerr << "Error: 3D-Value called for " << DIM << "dimensions \n";
            exit (EXIT_FAILURE);
        }

        Float y[DIM]; // make a duplicate
        y[0] = x[0]; y[1] = x[1]; y[2] = x[2]; 
        return value3d<DX, DY, DZ>(y);
    }

/*    template<int DX, int DY, int DZ>
    inline Float value(const Float x[]) const{
        Float y[DIM]; // make a duplicate
        for(int i=0; i<DIM; i++){
            y[i] = x[i];
        }
        if(DIM == 1)
            return value1d<DX>(y);
        if(DIM == 2)
            return value2d<DX, DY>(y);
        if(DIM == 3)
            return value3d<DX, DY, DZ>(y);
	    std::cerr << "Error: Current implementation of spline interpolation works only upto 3 dimensions \n";
        exit (EXIT_FAILURE);
    }
*/
private:
//public:
    Float xmin[DIM];
    Float xmax[DIM];
    Float xres[DIM];
    Float xres2[DIM];
    Float *coeff;
    int N[DIM];
    bool  built;
    Float z1;


    void convertToX(Float x[DIM]) const{
        for(int i=0; i<DIM; i++)
            x[i] = (x[i] - xmin[i]) * xres[i] ;
    }

    template<int D>
    Float kernel(Float x) const{
    	Float y = 0;
	    if(D == 0){
	        if(x > 2 || x < -2)
	            return y;
	        if(x > -2){
	            y += (x+2)*(x+2)*(x+2);
	            if(x > -1){
	                y += -4*(x+1)*(x+1)*(x+1);
	                if(x > 0){
	                    y += 6*x*x*x;
	                    if(x > 1){
	                        y += -4*(x-1)*(x-1)*(x-1);
	                    }
	                }
	            }
	        }
	        return y/6;
	    }else if(D == 1){
	        if(x > 2 || x < -2)
	            return y;
	        if(x > -2){
	            y += (x+2)*(x+2);
	            if(x > -1){
	                y += -4*(x+1)*(x+1);
	                if(x > 0){
	                    y += 6*x*x;
	                    if(x > 1){
	                        y += -4*(x-1)*(x-1);
	                    }
	                }
	            }
	        }
	        return y/2;
	    }else if(D == 2){
	        if(x > 2 || x < -2)
	            return y;
	        if(x > -2){
	            y += (x+2);
	            if(x > -1){
	                y += -4*(x+1);
	                if(x > 0){
	                    y += 6*x;
	                    if(x > 1){
	                        y += -4*(x-1);
	                    }
	                }
	            }
	        }
	        return y;
	    }else{
	        std::cerr << "Error: More than double derivative is not defined \n";
	    }
    }

    template<int DX>
    Float value1d(Float x[]) const{
        convertToX(x);
        Float v = 0;
        int wrap_index;
        for(int index = ceil(x[0]-2); index <= floor(x[0]+2); index++){
            wrap_index = modulo(index, 2*N[0]-2);
            if(wrap_index >= N[0]){
                wrap_index = 2*N[0] - 2 - wrap_index;
            }
            v += coeff[wrap_index] * kernel<DX>(x[0]-index);
        }
        if(DX == 1)
            v *= xres[0];
        if(DX == 2)
            v *= xres2[0];
        return v;
    }

    template<int DX, int DY>
    Float value2d(Float x[]) const{
        convertToX(x);
        Float v = 0;
        int wrap_index1, wrap_index2;
        for(int index1 = ceil(x[0]-2); index1 <= floor(x[0]+2); index1++){
            wrap_index1 = modulo(index1, 2*N[0]-2);
            if(wrap_index1 >= N[0]){
                wrap_index1 = 2*N[0] - 2 - wrap_index1;
            }
            for(int index2 = ceil(x[1]-2); index2 <= floor(x[1]+2); index2++){
                wrap_index2 = modulo(index2, 2*N[1]-2);
                if(wrap_index2 >= N[1]){
                    wrap_index2 = 2*N[1] - 2 - wrap_index2;
                }
                v += coeff[wrap_index1+wrap_index2*N[0]] * kernel<DX>(x[0]-index1) * kernel<DY>(x[1]-index2);
            }
        }
        if(DX == 1)
            v *= xres[0];
        if(DX == 2)
            v *= xres2[0];
        if(DY == 1)
            v *= xres[1];
        if(DY == 2)
            v *= xres2[1];
        return v;
    }

    template<int DX, int DY, int DZ>
    Float value3d(Float x[]) const{
        convertToX(x);
        Float v = 0;
        int wrap_index1, wrap_index2, wrap_index3;
        for(int index1 = ceil(x[0]-2); index1 <= floor(x[0]+2); index1++){
            wrap_index1 = modulo(index1, 2*N[0]-2);
            if(wrap_index1 >= N[0]){
                wrap_index1 = 2*N[0] - 2 - wrap_index1;
            }
            for(int index2 = ceil(x[1]-2); index2 <= floor(x[1]+2); index2++){
                wrap_index2 = modulo(index2, 2*N[1]-2);
                if(wrap_index2 >= N[1]){
                    wrap_index2 = 2*N[1] - 2 - wrap_index2;
                }
                for(int index3 = ceil(x[2]-2); index3 <= floor(x[2]+2); index3++){
                    wrap_index3 = modulo(index3, 2*N[2]-2);
                    if(wrap_index3 >= N[2]){
                        wrap_index3 = 2*N[2] - 2 - wrap_index3;
                    }
                    v += coeff[wrap_index1+wrap_index2*N[0]+wrap_index3*N[0]*N[1]] * kernel<DX>(x[0]-index1) * kernel<DY>(x[1]-index2) * kernel<DZ>(x[2]-index3);
                }
            }
        }
        if(DX == 1)
            v *= xres[0];
        if(DX == 2)
            v *= xres2[0];
        if(DY == 1)
            v *= xres[1];
        if(DY == 2)
            v *= xres2[1];
        if(DZ == 1)
            v *= xres[2];
        if(DZ == 2)
            v *= xres2[2];
        return v;
    }

    // build 1d is called from build2d and build3d so it is written in a generic style without affecting the contents of the class
    void build1d(const Float data[], const int &offset, const int &stride, const int &size, Float out[]) const{
        Float *cp = new Float[size];
        Float *cn = new Float[size];
        cp[0] = 0;

        for(int i=0; i<size; i++)
            cp[0] += data[offset+i*stride] * pow(z1, i);

        for(int i=size-2; i>0; i--)
            cp[0] += data[offset+i*stride] * pow(z1, 2*size-2-i);

        cp[0] /= (1-pow(z1, 2*size-2));

        for(int i=1; i<size; i++)
            cp[i] = data[offset + i*stride] + z1 * cp[i-1];


        cn[size-1] = z1/(z1*z1-1) * (cp[size-1] + z1*cp[size-2]); 

        for(int i=size-2; i>=0; i--)
            cn[i] = z1*(cn[i+1] - cp[i]);

        for(int i=0; i<size; i++)
            out[i] = 6*cn[i];


        delete[] cp;
        delete[] cn;
    }

    void build2d(const Float data[]){
        const int size = std::max(N[0], N[1]);
        Float *temp = new Float[size];

        // Build along rows
        for(int i=0; i<N[0]; i++){
            build1d(data, i, N[0], N[1], temp);
            for(int j=0; j<N[1]; j++){
                coeff[j*N[0]+i] = temp[j]; 
            }
        }

        // Build along columns
        for(int i=0; i<N[1]; i++){
            build1d(coeff, i*N[0], 1, N[0], temp);
            for(int j=0; j<N[0]; j++){
                coeff[i*N[0]+j] = temp[j]; 
            }
        }

        delete[] temp;
    }

    void build3d(const Float data[]){
        Float *temp = new Float[std::max(std::max(N[0], N[1]), N[2])];

        for(int k=0; k<N[2]; k++)
            for(int i=0; i<N[0]; i++){
                build1d(data, k*N[0]*N[1] + i, N[0], N[0], temp);
                for(int t=0; t < N[1]; t++)
                    coeff[i+t*N[0]+k*N[0]*N[1]] = temp[t];
            }

        for(int k=0; k<N[2]; k++)
            for(int j=0; j<N[1]; j++){
                build1d(coeff, k*N[0]*N[1] + j*N[0], 1, N[1], temp);
                for(int t=0; t < N[0]; t++)
                    coeff[t+j*N[0]+k*N[0]*N[1]] = temp[t];
            }

        for(int i=0; i<N[0]; i++)
            for(int j=0; j<N[1]; j++){
                build1d(coeff, j*N[0] + i, N[0]*N[1], N[2], temp);
                for(int t=0; t < N[2]; t++)
                    coeff[i+j*N[0]+t*N[0]*N[1]] = temp[t];
            }

        delete[] temp;
    }
};

