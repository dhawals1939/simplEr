/*
 * Spline.h
 * Interpolate 1D, 2D, 3D with basis splines
 *
 *  Created on: Dec 16, 2019
 *      Author: apedired
 */

#pragma once
#ifndef SPLINE_H_
#define SPLINE_H_

#include<string>
#include<fstream>
#include <iostream>
#include <constants.h>
#include <math.h>

namespace spline{

inline int modulo(int a, int b) {
    int r = a % b;
    return (r < 0) ? r+b : r;
}

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

namespace detail{

template<int D>
inline Float kernel(Float x);

template<>
inline Float kernel<0>(Float x){
    x = std::abs(x);
    if(x > 2)
        return (Float)(0.0);
    if(x > 1)
        return (ONE_SIXTH*(2-x)*(2-x)*(2-x));
    return (TWO_THIRD - x*x + HALF*x*x*x);

}
template<>
inline Float kernel<1>(Float x){
    int s = sgn(x);
    x = std::abs(x);
    if(x > 2)
        return (Float)(0.0);
    if(x > 1)
        return s*(-HALF*(2-x)*(2-x));
    return s*((1.5*x - 2)*x);

}
template<>
inline Float kernel<2>(Float x){
    x = std::abs(x);
    if(x > 2)
        return (Float)(0.0);
    if(x > 1)
        return 2-x;
    return 3*x-2;
}

}

template<int DIM>
class Spline{
public:

    Spline(){ // empty constructor, which waits for a proper initialization. HACK: FIXME
    }
    inline void initialize(const Float xmin[DIM], const Float xmax[DIM],  const int N[DIM]){ // to compensate for above HACK. FIXME;
        uint datasize = 1;
        for(int i=0; i < DIM; i++){
            this->xmin[i] = xmin[i];
            this->xmax[i] = xmax[i];
            this->N[i] = N[i];
            xres[i] = (N[i]-1)/(xmax[i] - xmin[i]);
            dxres[i] = xres[i];
            dxres2[i] = dxres[i] * dxres[i];
            datasize= datasize*N[i];
        }
        coeff = new Float[datasize];
        built = false; // Note: currently not testing when the value is asked as that would make code slow
        z1 = -2 + std::sqrt(3);
    }

    Spline(const std::string &rifgridFile){ // currently hardcoded for 3D and doesn't work for 1D, 2D
        Float *data;
        readVolAndBuild(rifgridFile);
    }

    Spline(const Float xmin[DIM], const Float xmax[DIM], const int N[DIM]){
        uint datasize = 1;
        for(int i=0; i < DIM; i++){
            this->xmin[i] = xmin[i];
            this->xmax[i] = xmax[i];
            this->N[i] = N[i];
            xres[i] = (N[i]-1)/(xmax[i] - xmin[i]);
            dxres[i] = xres[i];
            dxres2[i] = dxres[i] * dxres[i];
            datasize= datasize*N[i];
        }
        coeff = new Float[datasize];
        built = false; // Note: currently not testing when the value is asked as that would make code slow
        z1 = -2 + std::sqrt(3);
        
    }
    ~Spline(){
        delete[] coeff;
    }



    inline void readVolAndBuild(const std::string &filename){ // currently hardcoded for 3D and doesn't work for 1D, 2D
        float f;
        char c;
        int32_t i;
        std::ifstream fin;
        fin.open(filename, std::ios::binary);

        // Header should be VOL3
        fin.read(reinterpret_cast<char *>(&c), sizeof(c));if(c != 'V') {std::cerr << "file " << filename << " is not a VOL file. The prologue is not VOL" << std::endl; exit(-1);}
        fin.read(reinterpret_cast<char *>(&c), sizeof(c));if(c != 'O') {std::cerr << "file " << filename << " is not a VOL file. The prologue is not VOL" << std::endl; exit(-1);}
        fin.read(reinterpret_cast<char *>(&c), sizeof(c));if(c != 'L') {std::cerr << "file " << filename << " is not a VOL file. The prologue is not VOL" << std::endl; exit(-1);}
        fin.read(reinterpret_cast<char *>(&c), sizeof(c));if((int)c != 3) {std::cerr << "file " << filename << " is not of proper version (3). Instead it is " << int(c) << std::endl; exit(-1);}
        fin.read(reinterpret_cast<char *>(&i), sizeof(i));if(i != 1) {std::cerr << "type should be 1 (single-bit precision). Instead it is reported as " << i << std::endl; exit(-1);}

        // read the resolution.
        fin.read(reinterpret_cast<char *>(&i), sizeof(i)); N[0] = i;
        fin.read(reinterpret_cast<char *>(&i), sizeof(i)); N[1] = i;
        fin.read(reinterpret_cast<char *>(&i), sizeof(i)); N[2] = i;
        Float *data = new Float[N[0]*N[1]*N[2]];

        fin.read(reinterpret_cast<char *>(&i), sizeof(i));if(i != 1) {std::cerr << "should only be single channel. Currently reported as " << i << std::endl; exit(-1);}

        fin.read(reinterpret_cast<char *>(&f), sizeof(f)); xmin[0] = (Float) f;
        fin.read(reinterpret_cast<char *>(&f), sizeof(f)); xmin[1] = (Float) f;
        fin.read(reinterpret_cast<char *>(&f), sizeof(f)); xmin[2] = (Float) f;

        fin.read(reinterpret_cast<char *>(&f), sizeof(f)); xmax[0] = (Float) f;
        fin.read(reinterpret_cast<char *>(&f), sizeof(f)); xmax[1] = (Float) f;
        fin.read(reinterpret_cast<char *>(&f), sizeof(f)); xmax[2] = (Float) f;

        for(int z=0; z<N[2]; z++)
            for(int y=0; y<N[1]; y++)
                for(int x=0; x<N[0]; x++){
                    fin.read(reinterpret_cast<char *>(&f), sizeof(f));
                    data[x + N[0]*(y + z*N[1])] = (Float) f;
                }


        for(int i=0; i < DIM; i++){
            xres[i] = (N[i]-1)/(xmax[i] - xmin[i]);
            dxres[i] = xres[i];
            dxres2[i] = dxres[i] * dxres[i];
        }
        coeff = new Float[N[0]*N[1]*N[2]];
        built = false;
        z1 = -2 + std::sqrt(3);
        build3d(data);
    }

    inline void printcoeff3d() const{
        for(int k=0; k<N[2]; k++){
            for(int j=0; j<N[1]; j++){
                for(int i=0; i<N[0]; i++){
                    std::cout << coeff[i + j*N[0] + k*N[0]*N[1]] << ", ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }


    inline void build(const Float data[]){
        if(DIM == 1){
            build1d(data, 0, 1, N[0], coeff);
            built = true;
            return;
        }
        if(DIM == 2){
            build2d(data);
            built = true;
            return;
        }
        if(DIM == 3){
            build3d(data);
            built = true;
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

    // special functions that work only inside ERCRDR. comment outside
    inline Matrix3x3 hessian2d(const Float y[]) const{
        if(DIM != 2){
            std::cerr << "Error: 2D-Value called for " << DIM << "dimensions \n";
            exit (EXIT_FAILURE);
        }

        Float x[DIM]; // make a duplicate
        x[0] = y[0]; x[1] = y[1];

        Float Hxx = 0;
        Float Hxy = 0;
        Float Hyy = 0;

        convertToX(x);

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
                Hxx += coeff[wrap_index1+wrap_index2*N[0]] * kernel<2>(x[0]-index1) * kernel<0>(x[1]-index2);
                Hxy += coeff[wrap_index1+wrap_index2*N[0]] * kernel<1>(x[0]-index1) * kernel<1>(x[1]-index2);
                Hyy += coeff[wrap_index1+wrap_index2*N[0]] * kernel<0>(x[0]-index1) * kernel<2>(x[1]-index2);
            }
        }

        Hxx *= dxres2[0];
        Hyy *= dxres2[1];
        Hxy *= dxres[0]*dxres[1];

        return Matrix3x3(0, 0,   0,
                         0, Hyy, Hxy,
                         0, Hxy, Hxx);
    }

    inline tvec::Vec3f gradient2d(const Float y[]) const{
        if(DIM != 2){
            std::cerr << "Error: 2D-Value called for " << DIM << "dimensions \n";
            exit (EXIT_FAILURE);
        }

        Float x[DIM]; // make a duplicate
        x[0] = y[0]; x[1] = y[1];
        tvec::Vec3f v(0);

        convertToX(x);

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
                v.y += coeff[wrap_index1+wrap_index2*N[0]] * kernel<0>(x[0]-index1) * kernel<1>(x[1]-index2);
                v.z += coeff[wrap_index1+wrap_index2*N[0]] * kernel<1>(x[0]-index1) * kernel<0>(x[1]-index2);
            }
        }
        v.y *= dxres[1];
        v.z *= dxres[0];
        return v;
    }


    inline Float value(const Float x1[]) const{
        Float x[DIM]; // make a duplicate
        x[0] = x1[0]; x[1] = x1[1]; x[2] = x1[2];
        convertToX(x);
        Float v = 0;
        for(int index1 = ceil(x[0]-2); index1 <= floor(x[0]+2); index1++){
            for(int index2 = ceil(x[1]-2); index2 <= floor(x[1]+2); index2++){
                for(int index3 = ceil(x[2]-2); index3 <= floor(x[2]+2); index3++){
                    v += coeff[index1+index2*N[0]+index3*N[0]*N[1]] * kernel<0>(x[0]-index1) * kernel<0>(x[1]-index2) * kernel<0>(x[2]-index3);
                }
            }
        }
        return v;
    }


    inline tvec::Vec3f gradient(Float x[]) const{
        if(DIM != 3){
            std::cerr << "Error: 3D-Value called for " << DIM << "dimensions \n";
            exit (EXIT_FAILURE);
        }

        tvec::Vec3f v(0.0);

        convertToX(x);

        Float precomputeK0x;
        Float precomputeK0y;
        Float precomputeK0z;
        Float precomputeCoeff;

        for(int index1 = ceil(x[0]-2); index1 <= floor(x[0]+2); index1++){
            for(int index2 = ceil(x[1]-2); index2 <= floor(x[1]+2); index2++){
                for(int index3 = ceil(x[2]-2); index3 <= floor(x[2]+2); index3++){
                    precomputeCoeff = coeff[index1+index2*N[0]+index3*N[0]*N[1]];
                    precomputeK0x   = kernel<0>(x[0]-index1);
                    precomputeK0y   = kernel<0>(x[1]-index2);
                    precomputeK0z   = kernel<0>(x[2]-index3);
                    v.x += precomputeCoeff * kernel<1>(x[0]-index1) * precomputeK0y * precomputeK0z;
                    v.y += precomputeCoeff * precomputeK0x * kernel<1>(x[1]-index2) * precomputeK0z;
                    v.z += precomputeCoeff * precomputeK0x * precomputeK0y * kernel<1>(x[2]-index3);
                }
            }
        }
        v.x *= dxres[0];
        v.y *= dxres[1];
        v.z *= dxres[2];
        return v;
    }


    inline Matrix3x3 hessian(Float x[]) const{

        Float Hxx = 0;
        Float Hyy = 0;
        Float Hzz = 0;
        Float Hxy = 0;
        Float Hyz = 0;
        Float Hzx = 0;

        convertToX(x);

        Float precomputeK0x;
        Float precomputeK0y;
        Float precomputeK0z;
        Float precomputeK1x;
        Float precomputeK1y;
        Float precomputeK1z;
        Float precomputeCoeff;


        for(int index1 = ceil(x[0]-2); index1 <= floor(x[0]+2); index1++){
            for(int index2 = ceil(x[1]-2); index2 <= floor(x[1]+2); index2++){
                for(int index3 = ceil(x[2]-2); index3 <= floor(x[2]+2); index3++){
                    precomputeCoeff = coeff[index1+index2*N[0]+index3*N[0]*N[1]];
                    precomputeK0x   = kernel<0>(x[0]-index1);
                    precomputeK0y   = kernel<0>(x[1]-index2);
                    precomputeK0z   = kernel<0>(x[2]-index3);

                    precomputeK1x   = kernel<1>(x[0]-index1);
                    precomputeK1y   = kernel<1>(x[1]-index2);
                    precomputeK1z   = kernel<1>(x[2]-index3);

                    Hxx += precomputeCoeff * kernel<2>(x[0]-index1) * precomputeK0y * precomputeK0z;
                    Hyy += precomputeCoeff * precomputeK0x * kernel<2>(x[1]-index2) * precomputeK0z;
                    Hzz += precomputeCoeff * precomputeK0x * precomputeK0y * kernel<2>(x[2]-index3);

                    Hxy += precomputeCoeff * precomputeK1x * precomputeK1y * precomputeK0z;
                    Hyz += precomputeCoeff * precomputeK0x * precomputeK1y * precomputeK1z;
                    Hzx += precomputeCoeff * precomputeK1x * precomputeK0y * precomputeK1z;
                }
            }
        }

        Hxx *= dxres2[0];
        Hyy *= dxres2[1];
        Hzz *= dxres2[2];

        Hxy *= dxres[0]*dxres[1];
        Hyz *= dxres[1]*dxres[2];
        Hzx *= dxres[2]*dxres[0];

        return Matrix3x3(Hxx, Hxy, Hzx,
                         Hxy, Hyy, Hyz,
                         Hzx, Hyz, Hzz);
    }

    inline void valueAndGradient(Float x[], Float &f, tvec::Vec3f &v) const{

        v.x = v.y = v.z = 0;
        f = 0;

        convertToX(x);

        Float precomputeK0x;
        Float precomputeK0y;
        Float precomputeK0z;
        Float precomputeCoeff;


        for(int index1 = ceil(x[0]-2); index1 <= floor(x[0]+2); index1++){
            for(int index2 = ceil(x[1]-2); index2 <= floor(x[1]+2); index2++){
                for(int index3 = ceil(x[2]-2); index3 <= floor(x[2]+2); index3++){
                    precomputeCoeff = coeff[index1+index2*N[0]+index3*N[0]*N[1]];
                    precomputeK0x   = kernel<0>(x[0]-index1);
                    precomputeK0y   = kernel<0>(x[1]-index2);
                    precomputeK0z   = kernel<0>(x[2]-index3);

                    f   += precomputeCoeff * precomputeK0x * precomputeK0y * precomputeK0z;

                    v.x += precomputeCoeff * kernel<1>(x[0]-index1) * precomputeK0y * precomputeK0z;
                    v.y += precomputeCoeff * precomputeK0x * kernel<1>(x[1]-index2) * precomputeK0z;
                    v.z += precomputeCoeff * precomputeK0x * precomputeK0y * kernel<1>(x[2]-index3);
                }
            }
        }

        v.x *= dxres[0];
        v.y *= dxres[1];
        v.z *= dxres[2];
    }

    inline void gradientAndHessian(Float x[], tvec::Vec3f &v, Matrix3x3 &H) const{

        v.x = v.y = v.z = 0;

        Float Hxx = 0;
        Float Hyy = 0;
        Float Hzz = 0;
        Float Hxy = 0;
        Float Hyz = 0;
        Float Hzx = 0;

        convertToX(x);

        Float precomputeK0x;
        Float precomputeK0y;
        Float precomputeK0z;
        Float precomputeK1x;
        Float precomputeK1y;
        Float precomputeK1z;
        Float precomputeCoeff;


        for(int index1 = ceil(x[0]-2); index1 <= floor(x[0]+2); index1++){
            for(int index2 = ceil(x[1]-2); index2 <= floor(x[1]+2); index2++){
                for(int index3 = ceil(x[2]-2); index3 <= floor(x[2]+2); index3++){
                    precomputeCoeff = coeff[index1+index2*N[0]+index3*N[0]*N[1]];
                    precomputeK0x   = kernel<0>(x[0]-index1);
                    precomputeK0y   = kernel<0>(x[1]-index2);
                    precomputeK0z   = kernel<0>(x[2]-index3);

                    precomputeK1x   = kernel<1>(x[0]-index1);
                    precomputeK1y   = kernel<1>(x[1]-index2);
                    precomputeK1z   = kernel<1>(x[2]-index3);

                    Hxx += precomputeCoeff * kernel<2>(x[0]-index1) * precomputeK0y * precomputeK0z;
                    Hyy += precomputeCoeff * precomputeK0x * kernel<2>(x[1]-index2) * precomputeK0z;
                    Hzz += precomputeCoeff * precomputeK0x * precomputeK0y * kernel<2>(x[2]-index3);

                    Hxy += precomputeCoeff * precomputeK1x * precomputeK1y * precomputeK0z;
                    Hyz += precomputeCoeff * precomputeK0x * precomputeK1y * precomputeK1z;
                    Hzx += precomputeCoeff * precomputeK1x * precomputeK0y * precomputeK1z;

                    v.x += precomputeCoeff * precomputeK1x * precomputeK0y * precomputeK0z;
                    v.y += precomputeCoeff * precomputeK0x * precomputeK1y * precomputeK0z;
                    v.z += precomputeCoeff * precomputeK0x * precomputeK0y * precomputeK1z;
                }
            }
        }

        v.x *= dxres[0];
        v.y *= dxres[1];
        v.z *= dxres[2];

        Hxx *= dxres2[0];
        Hyy *= dxres2[1];
        Hzz *= dxres2[2];

        Hxy *= dxres[0]*dxres[1];
        Hyz *= dxres[1]*dxres[2];
        Hzx *= dxres[2]*dxres[0];

        H = Matrix3x3(Hxx, Hxy, Hzx,
                       Hxy, Hyy, Hyz,
                       Hzx, Hyz, Hzz);
    }

    inline void valueGradientAndHessian(Float x[], Float &f, tvec::Vec3f &v, Matrix3x3 &H) const{

        v.x = v.y = v.z = 0;
        f = 0;

        Float Hxx = 0;
        Float Hyy = 0;
        Float Hzz = 0;
        Float Hxy = 0;
        Float Hyz = 0;
        Float Hzx = 0;

        convertToX(x);

        Float precomputeK0x;
        Float precomputeK0y;
        Float precomputeK0z;
        Float precomputeK1x;
        Float precomputeK1y;
        Float precomputeK1z;
        Float precomputeCoeff;


        for(int index1 = ceil(x[0]-2); index1 <= floor(x[0]+2); index1++){
            for(int index2 = ceil(x[1]-2); index2 <= floor(x[1]+2); index2++){
                for(int index3 = ceil(x[2]-2); index3 <= floor(x[2]+2); index3++){
                    precomputeCoeff = coeff[index1+index2*N[0]+index3*N[0]*N[1]];
                    precomputeK0x   = kernel<0>(x[0]-index1);
                    precomputeK0y   = kernel<0>(x[1]-index2);
                    precomputeK0z   = kernel<0>(x[2]-index3);

                    precomputeK1x   = kernel<1>(x[0]-index1);
                    precomputeK1y   = kernel<1>(x[1]-index2);
                    precomputeK1z   = kernel<1>(x[2]-index3);

                    f += precomputeCoeff * precomputeK0x * precomputeK0y * precomputeK0z;

                    Hxx += precomputeCoeff * kernel<2>(x[0]-index1) * precomputeK0y * precomputeK0z;
                    Hyy += precomputeCoeff * precomputeK0x * kernel<2>(x[1]-index2) * precomputeK0z;
                    Hzz += precomputeCoeff * precomputeK0x * precomputeK0y * kernel<2>(x[2]-index3);

                    Hxy += precomputeCoeff * precomputeK1x * precomputeK1y * precomputeK0z;
                    Hyz += precomputeCoeff * precomputeK0x * precomputeK1y * precomputeK1z;
                    Hzx += precomputeCoeff * precomputeK1x * precomputeK0y * precomputeK1z;

                    v.x += precomputeCoeff * precomputeK1x * precomputeK0y * precomputeK0z;
                    v.y += precomputeCoeff * precomputeK0x * precomputeK1y * precomputeK0z;
                    v.z += precomputeCoeff * precomputeK0x * precomputeK0y * precomputeK1z;
                }
            }
        }

        v.x *= dxres[0];
        v.y *= dxres[1];
        v.z *= dxres[2];

        Hxx *= dxres2[0];
        Hyy *= dxres2[1];
        Hzz *= dxres2[2];

        Hxy *= dxres[0]*dxres[1];
        Hyz *= dxres[1]*dxres[2];
        Hzx *= dxres[2]*dxres[0];

        H = Matrix3x3(Hxx, Hxy, Hzx,
                       Hxy, Hyy, Hyz,
                       Hzx, Hyz, Hzz);
    }

    inline Float getStride(int dim) const{
        return 1.0/xres[dim];
    }
private:
    Float xmin[DIM];
    Float xmax[DIM];
    Float xres[DIM];
    Float dxres[DIM];
    Float dxres2[DIM];
    Float *coeff;
    int N[DIM];
    bool  built;
    Float z1;


    inline void convertToX(Float x[DIM]) const{
        for(int i=0; i<DIM; i++)
            x[i] = (x[i] - xmin[i]) * xres[i] ;
    }


    template<int D>
    inline Float kernel(Float x) const{
        return detail::kernel<D>(x);
    }

    template<int DX>
    inline Float value1d(Float x[]) const{
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
            v *= dxres[0];
        if(DX == 2)
            v *= dxres2[0];
        return v;
    }

    template<int DX, int DY>
    inline Float value2d(Float x[]) const{
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
            v *= dxres[0];
        if(DX == 2)
            v *= dxres2[0];
        if(DY == 1)
            v *= dxres[1];
        if(DY == 2)
            v *= dxres2[1];
        return v;
    }

    template<int DX, int DY, int DZ>
    inline Float value3d(Float x[]) const{
        convertToX(x);
        Float v = 0;
        for(int index1 = ceil(x[0]-2); index1 <= floor(x[0]+2); index1++){
            for(int index2 = ceil(x[1]-2); index2 <= floor(x[1]+2); index2++){
                for(int index3 = ceil(x[2]-2); index3 <= floor(x[2]+2); index3++){
                    v += coeff[index1+index2*N[0]+index3*N[0]*N[1]] * kernel<DX>(x[0]-index1) * kernel<DY>(x[1]-index2) * kernel<DZ>(x[2]-index3);
                }
            }
        }
        if(DX == 1)
            v *= dxres[0];
        if(DX == 2)
            v *= dxres2[0];
        if(DY == 1)
            v *= dxres[1];
        if(DY == 2)
            v *= dxres2[1];
        if(DZ == 1)
            v *= dxres[2];
        if(DZ == 2)
            v *= dxres2[2];
        return v;
    }

    // build 1d is called from build2d and build3d so it is written in a generic style without affecting the contents of the class
    inline void build1d(const Float data[], const int &offset, const int &stride, const int &size, Float out[]) const{
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

    inline void build2d(const Float data[]){
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

    inline void build3d(const Float data[]){
        Float *temp = new Float[std::max(std::max(N[0], N[1]), N[2])];

        for(int k=0; k<N[2]; k++)
            for(int i=0; i<N[0]; i++){
                build1d(data, k*N[0]*N[1] + i, N[0], N[1], temp);
                for(int t=0; t < N[1]; t++)
                    coeff[i+t*N[0]+k*N[0]*N[1]] = temp[t];
            }

        for(int k=0; k<N[2]; k++)
            for(int j=0; j<N[1]; j++){
                build1d(coeff, k*N[0]*N[1] + j*N[0], 1, N[0], temp);
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

}   /* namespace spline */

#endif /* SPLINE_H_ */


