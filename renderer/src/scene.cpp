/*
 * scene.cpp
 *
 *  Created on: Nov 26, 2015
 *      Author: igkiou
 */

#include "scene.h"
#include "util.h"
#include <iostream>
#include "math.h"
#include <boost/math/special_functions.hpp>

#include <bits/stdc++.h>
#include <chrono>

namespace scn {


void sampleRandomDirection(tvec::Vec2f &randDirection,  smp::Sampler &sampler){ // Need to find a better place for these
	Float phi= 2*M_PI*sampler();
	randDirection.x = std::cos(phi);
	randDirection.y = std::sin(phi);
}

void sampleRandomDirection(tvec::Vec3f &randDirection, smp::Sampler &sampler){
//	Float phi= 2*M_PI*sampler();
//	Float r1 = sampler();
//	Float sintheta = std::sqrt(r1*(2-r1));
//	randDirection.x = 1-r1;
//	randDirection.y = sintheta*std::cos(phi);
//	randDirection.z = sintheta*std::sin(phi);
	randDirection = warp::squareToUniformHemisphere(tvec::Vec2f(sampler(), sampler())); // this sampling is done in z=1 direction. need     to compensate for it.
//	invPDF = 2.0f * M_PI;
	Float temp = randDirection.x;
	randDirection.x =-randDirection.z; // compensating that the direction of photon propagation is -x
	randDirection.z = randDirection.y;
	randDirection.y = temp;

}



/*
 * Returns true if it is possible to intersect, and false otherwise.
 * disx is the smallest distance that must be traveled till intersection if outside the box. If inside the box, dis.x = 0;
 * disy is the smallest distance that must be traveled till intersection if inside the box.
 */
template <template <typename> class VectorType>
bool Block<VectorType>::intersect(const VectorType<Float> &p, const VectorType<Float> &d, Float &disx, Float &disy) const {

	VectorType<Float> l(M_MIN), r(M_MAX);
	for (int i = 0; i < p.dim; ++i) {
		if (std::abs(d[i]) > M_EPSILON * std::max(FPCONST(1.0), std::abs(d[i]))) {
			l[i] = (m_blockL[i] - p[i])/d[i];
			r[i] = (m_blockR[i] - p[i])/d[i];
			if (l[i] > r[i]) {
				std::swap(l[i], r[i]);
			}
		} else if (m_blockL[i] > p[i] || p[i] > m_blockR[i]) {
			return false;
		}
	}

	disx = l.max();
	disy = r.min();
	if (disx < FPCONST(0.0)) {
		disx = FPCONST(0.0);
	}
	if (disx - disy > -M_EPSILON) {
		return false;
	}

	return true;
}

template <template <typename> class VectorType>
bool Camera<VectorType>::samplePosition(VectorType<Float> &pos, smp::Sampler &sampler) const {
	pos = m_origin;
	for (int iter = 1; iter < m_origin.dim; ++iter) {
		pos[iter] += - m_plane[iter - 1] / FPCONST(2.0) + sampler() * m_plane[iter - 1];
	}
	return true;
}

template <template <typename> class VectorType>
bool AreaSource<VectorType>::sampleRay(VectorType<Float> &pos, VectorType<Float> &dir, smp::Sampler &sampler) const {
	pos = m_origin;
	for (int iter = 1; iter < m_origin.dim; ++iter) {
		pos[iter] += - m_plane[iter - 1] / FPCONST(2.0) + sampler() * m_plane[iter - 1];
	}
	dir = m_dir;

	return true;
}

template <template <typename> class VectorType>
bool AreaTexturedSource<VectorType>::sampleRay(VectorType<Float> &pos, VectorType<Float> &dir, smp::Sampler &sampler, Float &totalDistance) const {
	pos = m_origin;

	// sample pixel position first
	int pixel = m_textureSampler.sample(sampler());
	int p[2];
	m_texture.ind2sub(pixel, p[0], p[1]);

	// Now find a random location on the pixel
	for (int iter = 1; iter < m_origin.dim; ++iter) {
		pos[iter] += - m_plane[iter - 1] / FPCONST(2.0) + p[iter - 1] * m_pixelsize[iter-1] + sampler() * m_pixelsize[iter - 1];
	}

	dir = m_dir;

	//FIXME: Hack: Works only for m_dir = [-1 0 0]
	Float z   = sampler()*(1-m_ct) + m_ct;
	Float zt  = std::sqrt(1-z*z);
	Float phi = sampler()*2*M_PI;
	dir[0] = -z;
	dir[1] = zt*std::cos(phi);
	dir[2] = zt*std::sin(phi);

	return propagateTillMedium(pos, dir, totalDistance);

//	std::cout << dir[0] << ", " << dir[1] << ", " << dir[2] << std::endl;

//	if(m_emittertype != EmitterType::directional)
//		std::cout << "Diffuse source not implemented; only directional source is implemented";
}

template <template <typename> class VectorType>
double US<VectorType>::bessel_RIF(const VectorType<Float> &p, const Float &scaling) const{
    VectorType<Float> p_axis = p_u + dot(p - p_u , axis_uz)*axis_uz; // point on the axis closest to p

    Float r    = (p-p_axis).length();
    Float dotp = dot(p-p_axis, axis_ux);
    Float detp = dot(cross(axis_ux, p-p_axis), axis_uz);
    Float phi  = std::atan2(detp, dotp);
//    Float r   = std::sqrt(p.x*p.x + p.y*p.y);
//    Float phi = std::atan2(p.y, p.x);

	return n_o + n_max * scaling * jn(mode, k_r*r) * std::cos(mode*phi);
}

template <template <typename> class VectorType>
const VectorType<Float> US<VectorType>::bessel_dRIF(const VectorType<Float> &q, const Float &scaling) const{

    VectorType<Float> p_axis = p_u + dot(q - p_u, axis_uz)*axis_uz; // point on the axis closest to p

    VectorType<Float> p      = q - p_axis; // acts like p in case of axis aligned

    Float r    = p.length();
    Float dotp = dot(p, axis_ux);
    Float detp = dot(cross(axis_ux, p), axis_uz);
    Float phi  = std::atan2(detp, dotp);

    if(r < M_EPSILON)
    	p.y = M_EPSILON;
    if(r < M_EPSILON)
    	p.z = M_EPSILON;
    if(r < M_EPSILON)
    	r = M_EPSILON;

    Float krr = k_r * r;

//    Float besselj   = boost::math::cyl_bessel_j(mode, krr);
//
//    Float dbesselj  = mode/(krr) * besselj - boost::math::cyl_bessel_j(mode+1, krr);
//
    Float besselj   = jn(mode, krr);

    Float dbesselj  = mode/(krr) * besselj - jn(mode+1, krr);

    Float invr  = 1.0/r;
    Float invr2 = invr * invr;

    Float cosmp  = std::cos(mode * phi);
    Float sinmp  = std::sin(mode * phi);

    VectorType<Float> dn(0.0,
            			 n_max * scaling * (dbesselj * k_r * p.y * invr * cosmp - besselj*mode*sinmp*p.z*invr2),
            			 n_max * scaling * (dbesselj * k_r * p.z * invr * cosmp + besselj*mode*sinmp*p.y*invr2)
                         );
//    VectorType<Float> dn(0.0,
//    					 n_max * (dbesselj * k_r * p.y * invr),
//                         n_max * (dbesselj * k_r * p.z * invr)
//                         ); // Adithya: FIXME: Assumed for now that ray is traveling in x direction.
    return dn;
}

template <template <typename> class VectorType>
const Matrix3x3 US<VectorType>::bessel_HessianRIF(const VectorType<Float> &q, const Float &scaling) const{
    VectorType<Float> p_axis = p_u + dot(q - p_u, axis_uz)*axis_uz; // point on the axis closest to p
    VectorType<Float> p      = q - p_axis; // acts like p in case of axis aligned

    Float r    = p.length();
    Float dotp = dot(p, axis_ux);
    Float detp = dot(cross(axis_ux, p), axis_uz);
    Float phi  = std::atan2(detp, dotp);

    if(r < M_EPSILON)
    	p.y = M_EPSILON; // equivalent of y in matlab
    if(r < M_EPSILON)
    	p.z = M_EPSILON; // equivalent of x in matlab
    if(r < M_EPSILON)
    	r = M_EPSILON;

    Float krr = k_r * r;

    Float nbesselj  = jn(mode    , krr);
    Float nbesselj1 = jn(mode + 1, krr);
    Float nbesselj2 = jn(mode + 2, krr);

    Float dbesselj  =     mode/(krr) * nbesselj  - nbesselj1;
    Float dbesselj1 = (mode+1)/(krr) * nbesselj1 - nbesselj2;

    Float invr  = 1.0/r;
    Float invr2 = invr * invr;

    Float cosp   = std::cos(phi);
    Float sinp   = std::sin(phi);
    Float cosmp  = std::cos(mode * phi);
    Float sinmp  = std::sin(mode * phi);
    Float cosm1p = std::cos((mode-1) * phi);
    Float sinm1p = std::sin((mode-1) * phi);
    Float cosm2p = std::cos((mode-2) * phi);
    Float sinm2p = std::sin((mode-2) * phi);

    Float Hxx = n_max * (
						+ nbesselj  * mode * invr2 * (-cosm2p + mode * sinm1p * sinp)
						+ dbesselj  * mode * invr * k_r * cosm1p * cosp
						- nbesselj1 * k_r * p.y * invr2 * (sinp * cosmp + mode * cosp * sinmp)
						- dbesselj1 * k_r * k_r * cosp * cosp * cosmp
    					);
    Float Hxy = n_max * (
						- nbesselj  * mode * invr2 * (-sinm2p + mode * sinm1p * cosp)
						+ dbesselj  * mode * invr * k_r * cosm1p * sinp
						+ nbesselj1 * k_r * p.z * invr2 * (sinp * cosmp + mode * cosp * sinmp)
						- dbesselj1 * k_r * k_r * cosp * sinp * cosmp
    					);
    Float Hyy = -n_max * (
						+ nbesselj  * mode * invr2 * (-cosm2p + mode * cosm1p * cosp)
						+ dbesselj  * mode * invr * k_r * sinm1p * sinp
						+ nbesselj1 * k_r * p.z * invr2 * (cosp * cosmp - mode * sinp * sinmp)
						+ dbesselj1 * k_r * k_r * sinp * sinp * cosmp
    					);
    return Matrix3x3(0, 0,   0,
    				 0, Hyy, Hxy,
					 0, Hxy, Hxx);


}


template <template <typename> class VectorType>
void Scene<VectorType>::er_step(VectorType<Float> &p, VectorType<Float> &d, const Float &stepSize, const Float &scaling) const{
#ifndef OMEGA_TRACKING
    d += HALF * stepSize * dV(p, d, scaling);
    p +=        stepSize * d/m_us.RIF(p, scaling);
    d += HALF * stepSize * dV(p, d, scaling);
#else
    Float two = 2; // To avoid type conversion

    VectorType<Float> K1P = stepSize * dP(d);
    VectorType<Float> K1O = stepSize * dOmega(p, d);

    VectorType<Float> K2P = stepSize * dP(d + HALF*K1O);
    VectorType<Float> K2O = stepSize * dOmega(p + HALF*K1P, d + HALF*K1O);

    VectorType<Float> K3P = stepSize * dP(d + HALF*K2O);
    VectorType<Float> K3O = stepSize * dOmega(p + HALF*K2P, d + HALF*K2O);

    VectorType<Float> K4P = stepSize * dP(d + K3O);
    VectorType<Float> K4O = stepSize * dOmega(p + K3P, d + K3O);

    p = p + ONE_SIXTH * (K1P + two*K2P + two*K3P + K4P);
    d = d + ONE_SIXTH * (K1O + two*K2O + two*K3O + K4O);
#endif
}

template <template <typename> class VectorType>
void Scene<VectorType>::er_derivativestep(VectorType<Float> &p, VectorType<Float> &v, Matrix3x3 &dpdv0, Matrix3x3 &dvdv0, const Float &stepSize, const Float &scaling) const{
    v 	  += HALF * stepSize * dV(p, v, scaling);
    dvdv0 += HALF * stepSize * d2V(p, v, dpdv0, scaling);
    p 	  +=        stepSize * v/m_us.RIF(p, scaling);
    dpdv0 +=        stepSize * d2Path(p, v, dpdv0, dvdv0, scaling);
    v	  += HALF * stepSize * dV(p, v, scaling);
    dvdv0 += HALF * stepSize * d2V(p, v, dpdv0, scaling);
}

//template <template <typename> class VectorType>
//void Scene<VectorType>::er_derivativestep(VectorType<Float> &p, VectorType<Float> &d, const Float &stepSize, const Float &scaling) const{
//#ifndef OMEGA_TRACKING
//    d += HALF * stepSize * dV(p, d, scaling);
//    p +=        stepSize * d/m_us.RIF(p, scaling);
//    d += HALF * stepSize * dV(p, d, scaling);
//#else
//    std::err << "Non omega derivative tracking not implemented;" << std::endl
//#endif
//}

template <template <typename> class VectorType>
void Scene<VectorType>::trace(VectorType<Float> &p, VectorType<Float> &d, const Float &dist, const Float &scaling) const{
    Float distance = dist;
    int steps = distance/m_us.er_stepsize;
    distance  = distance - steps * m_us.er_stepsize;
    for(int i = 0; i < steps; i++)
        er_step(p, d, m_us.er_stepsize, scaling);
    er_step(p, d, distance, scaling);
}

// ADI: CONVERT THIS TO A BISECTION SEARCH
/* This code is similar to trace code and the intersect code of "Block" structure*/
/* Based on ER equations, the ray is traced till we either meet end of a block or the distance (in that case, we have an error) */
template <template <typename> class VectorType>
void Scene<VectorType>::traceTillBlock(VectorType<Float> &p, VectorType<Float> &d, const Float &dist, Float &disx, Float &disy, Float &totalOpticalDistance, const Float &scaling) const{

	VectorType<Float> oldp, oldd;

    Float distance = 0;
    long int maxsteps = dist/m_us.er_stepsize + 1, i, precision = m_us.getPrecision();

    Float current_stepsize = m_us.er_stepsize;

    for(i = 0; i < maxsteps; i++){
    	oldp = p;
    	oldd = d;

    	er_step(p, d, current_stepsize, scaling);

    	// check if we are at the intersection or crossing the sampled dist, then, estimate the distance and keep going more accurately towards the boundary or sampled dist
    	if(!m_block.inside(p) || (distance + current_stepsize) > dist){
    		precision--;
    		if(precision < 0)
    			break;
    		p = oldp;
    		d = oldd;
    		current_stepsize = current_stepsize / 10;
    		i  = 0;
    		maxsteps = 11;
    	}else{
    		distance += current_stepsize;
#if !USE_SIMPLIFIED_TIMING
    		totalOpticalDistance += current_stepsize * m_us.RIF(p, scaling);
#endif
    	}
    }

    Assert(i < maxsteps);
    disx = 0;
    disy = distance;

}

template <template <typename> class VectorType>
void Scene<VectorType>::trace_optical_distance(VectorType<Float> &p, VectorType<Float> &d, const Float &dist, const Float &scaling) const{
    Float distance = dist;
    int maxSteps = distance/m_us.er_stepsize + 1;
    Float opticalPathLength = 0;
    VectorType<Float> oldp;
    VectorType<Float> oldd;
    for(int i = 0; i < maxSteps; i++){
        oldp = p;
        oldd = d;
        er_step(p, d, m_us.er_stepsize, scaling);
        opticalPathLength += m_us.er_stepsize * m_us.RIF(HALF * (oldp + p), scaling);
        if(opticalPathLength > distance){
            p = oldp;
            d = oldd;
            break;
        }
    }
    distance = (distance - opticalPathLength)/m_us.RIF(p, scaling);
    er_step(p, d, distance, scaling);
}

template <template <typename> class VectorType>

bool Scene<VectorType>::makeSurfaceDirectConnection(const VectorType<Float> &p1, const VectorType<Float> &p2, const Float &scaling, smp::Sampler &sampler,
														Float &distTravelled, VectorType<Float> &dirToSensor, Float &distToSensor, Float &weight,
														scn::NEECostFunction<VectorType> &costFunction, Problem &problem, Float *initialization) const{

	Matrix3x3 dpdv0((Float)0);
	Matrix3x3 dvdv0((Float)1, 0, 0,
					0, 1, 0,
					0, 0, 1);

	while(true){
		VectorType<Float> v;


#ifdef PRINT_DEBUGLOG
		v = p2 - p1; // Hack to make direct connections match the crdr.
		v.normalize();
#else
		if(m_us.m_useInitializationHack){
			v = p2 - p1; // Hack to make direct connections match the crdr.
			v.normalize();
		}
		else
			sampleRandomDirection(v, sampler);
#endif

//		CostFunction* cost_function = new NEECostFunction<tvec::TVector3>(this, p1, p2, dpdv0, dvdv0, scaling);
//		Problem problem;
//		double x[] = {v.x, v.y, v.z};
//		problem.AddResidualBlock(cost_function, NULL, x);

		costFunction.updateParameters(p1, p2, dpdv0, dvdv0, scaling);

		initialization[0] = v.x; initialization[1] = v.y; initialization[2] = v.z;

		Solver::Summary summary;
		Solve(m_options, &problem, &summary);

		if(summary.final_cost < m_us.getTol2()){
			dirToSensor[0] = initialization[0];
			dirToSensor[1] = initialization[1];
			dirToSensor[2] = initialization[2];
			dirToSensor.normalize();
			break;
		}

		// Did not converge, so perform russian roulette
		if(sampler() < m_us.getrrWeight())
			weight = weight * m_us.getInvrrWeight();
		else{
			dirToSensor[0] = initialization[0];
			dirToSensor[1] = initialization[1];
			dirToSensor[2] = initialization[2];
			dirToSensor.normalize();
			return false;
		}
	}
	// success, the algorithm found a solution.
	VectorType<Float> error;
	Matrix3x3 derror;
	Float opticalPathLength;
	VectorType<Float> v = dirToSensor * m_us.RIF(p1, scaling);

	computePathLengthstillZ(v, p1, p2, opticalPathLength, distToSensor, scaling);
#if !USE_SIMPLIFIED_TIMING
	distTravelled += opticalPathLength;
#endif
	return true;
}

template <template <typename> class VectorType>
void Scene<VectorType>::computePathLengthstillZ(const VectorType<Float> &v_i, const VectorType<Float> &p1, const VectorType<Float> &p2, Float &opticalPathLength, Float &t_l, const Float &scaling) const{

	if(v_i.x > 0){ // surely failing case
		t_l = M_MAX;
		opticalPathLength = M_MAX;
	}
//
//#ifdef PRINT_DEBUGLOG
//	std::cout << "Trying to connect: " << std::endl;
//	std::cout << "P1: (" << p1.x  << ", " << p1.y  << ", " << p1.z  << "); " << std::endl;
//	std::cout << "P2: (" << p2.x  << ", " << p2.y  << ", " << p2.z  << "); " << std::endl;
//	std::cout << "Vi: (" << v_i.x << ", " << v_i.y << ", " << v_i.z << "); " << std::endl;
//#endif

	t_l = 0 ; // t_l is geometric length, required for computation of the radiance

	Float currentStepSize = m_us.getStepSize();
	int maxSteps = 1e5;
	opticalPathLength = 0;
	int dec_precision = m_us.getPrecision();
	int nBisectionSearches = ceil(dec_precision/log10(2)); //ADI: Make a log10(2) constant

	VectorType<Float> p = p1, oldp = p1, v = v_i, oldv;
	v.normalize();
	v = v*m_us.RIF(p, scaling);
	oldv = v;

	for(int i = 0; i < maxSteps; i++){
		oldp     = p;
		oldv     = v;
		er_step(p, v, currentStepSize, scaling);
		if(p.x < p2.x){ // outside medium (ADI: FIXME: Direct is negative. Hardcoded now :()
			while(nBisectionSearches > 0){
				nBisectionSearches--;
				p     = oldp    ;
				v     = oldv    ;
				currentStepSize = currentStepSize/2;
				er_step(p, v, currentStepSize, scaling);
				if(p.x > p2.x){
					t_l += currentStepSize;
					oldp     = p;
					oldv     = v;
					opticalPathLength += currentStepSize*m_us.RIF(HALF * (oldp + p), scaling);
				}
			}
			break;
		}else{
			t_l += currentStepSize;
			opticalPathLength += currentStepSize*m_us.RIF(HALF * (oldp + p), scaling);
		}
	}

//	if( (p2 - p).lengthSquared() > m_us.getTol2()){
//		std::cout << "Gradient descent converged but the direct is diverging";
//		exit(-1);
//	}

//#ifdef PRINT_DEBUGLOG
//	std::cout << "Geometric length: " << t_l << std::endl;
//#endif
}

template <template <typename> class VectorType>
void Scene<VectorType>::computefdfNEE(const VectorType<Float> &v_i, const VectorType<Float> &p1, const VectorType<Float> &p2, Matrix3x3 &dpdv0, Matrix3x3 &dvdv0, const Float &scaling, VectorType<Float> &error, Matrix3x3 &derror) const{

	if(v_i.x > 0){ // surely failing case
		error = VectorType<Float>(0);
		derror = Matrix3x3((Float)0);
	}

	Float currentStepSize = m_us.getStepSize();
	int maxSteps = 1e5;
	int dec_precision = m_us.getPrecision();
	int nBisectionSearches = ceil(dec_precision/log10(2)); //ADI: Make a log10(2) constant

	VectorType<Float> p = p1, oldp = p1, v = v_i, oldv = v;
	Matrix3x3 olddpdv0 = dpdv0, olddvdv0 = dvdv0;


	// Normalize v_i
    Float r = m_us.RIF(p, scaling);
    Float n1 = v.length();
    Float n2 = n1*n1;
    Float n3 = n2*n1;
    dvdv0 = (n2*Matrix3x3(1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0) - Matrix3x3(v, v))/n3*r*dvdv0;
    v = v/n1 * r;


	for(int i = 0; i < maxSteps; i++){
		oldp     = p;
		oldv     = v;
		olddpdv0 = dpdv0;
		olddvdv0 = dvdv0;
		er_derivativestep(p, v, dpdv0, dvdv0, currentStepSize, scaling);
		if(p.x < p2.x){ // outside medium (ADI: FIXME: Direct is negative. Hardcoded now :()
			while(nBisectionSearches > 0){
				nBisectionSearches--;
				p     = oldp    ;
				v     = oldv    ;
				dpdv0 = olddpdv0;
				dvdv0 = olddvdv0;
				currentStepSize = currentStepSize/2;
				er_derivativestep(p, v, dpdv0, dvdv0, currentStepSize, scaling);
				if(p.x > p2.x){
					oldp     = p;
					oldv     = v;
					olddpdv0 = dpdv0;
					olddvdv0 = dvdv0;
				}
			}
			break;
		}
	}

	VectorType<Float> dpdt = v/m_us.RIF(p, scaling);
	VectorType<Float> dgdQ(-1, 0, 0);

	Float dp = dot(dgdQ, dpdt);
    Matrix3x3 dQdV = (Matrix3x3(1, 0, 0, 0, 1, 0, 0, 0, 1) - Matrix3x3(dpdt, dgdQ)/dp)*dpdv0, dQdVt;
    dQdV.transpose(dQdVt);
//    error  = HALF * (p - p2).lengthSquared();
//    derror =  dQdVt*(p - p2);
    error  = (p - p2);
    derror =  dQdVt;
}

template <template <typename> class VectorType>
bool Scene<VectorType>::genRay(VectorType<Float> &pos, VectorType<Float> &dir,
						smp::Sampler &sampler, Float &totalDistance) const {

	if (m_source.sampleRay(pos, dir, sampler, totalDistance)) {
//		Float dist = FPCONST(0.0);
//		Assert(std::abs(dir.x) >= M_EPSILON);
//		if (dir.x >= M_EPSILON) {
//			dist = (m_mediumBlock.getBlockL().x - pos.x) / dir.x;
//		} else if (dir.x <= -M_EPSILON) {
//			dist = (m_mediumBlock.getBlockR().x - pos.x) / dir.x;
//		}
//		pos += dist * dir;
//		pos.x += M_EPSILON * 2;
//		dir = m_refrDir;
		return true;
	} else {
		return false;
	}
}

template <template <typename> class VectorType>
bool Scene<VectorType>::genRay(VectorType<Float> &pos, VectorType<Float> &dir,
						smp::Sampler &sampler,
						VectorType<Float> &possrc, VectorType<Float> &dirsrc,
						Float &totalDistance) const {

	if (m_source.sampleRay(pos, dir, sampler, totalDistance)) {
		possrc = pos;
		dirsrc = dir;
//		Float dist = FPCONST(0.0);
//		Assert(std::abs(dir.x) >= M_EPSILON);
//		if (dir.x >= M_EPSILON) {
//			dist = (m_mediumBlock.getBlockL().x - pos.x) / dir.x;
//		} else if (dir.x <= -M_EPSILON) {
//			dist = (m_mediumBlock.getBlockR().x - pos.x) / dir.x;
//		}
//		pos += dist * dir;
		dir = m_refrDir;
		return true;
	} else {
		return false;
	}
}


template <template <typename> class VectorType>
bool Scene<VectorType>::movePhotonTillSensor(VectorType<Float> &p, VectorType<Float> &d, Float &distToSensor, Float &totalOpticalDistance,
									smp::Sampler &sampler, const Float& scaling) const {
	// moveTillSensor: moves the photon and reflects (with probability) and keeps going till it reaches sensor. TODO: change to weight

	Float LargeDist = (Float) 10000;

	Float disx, disy;
	VectorType<Float> d1, norm;
	traceTillBlock(p, d, LargeDist, disx, disy, totalOpticalDistance, scaling);
	distToSensor = disy;
	LargeDist -= disy;
	while(true){
		if(LargeDist < 0){
			std::cout << "Error in movePhotonTillSensorCode; Large distance is not large enough" << std::endl;
			return false;
		}
		int i;
		norm.zero();
		for (i = 0; i < p.dim; ++i) {
			if (std::abs(m_block.getBlockL()[i] - p[i]) < 2*M_EPSILON) {
				norm[i] = -FPCONST(1.0);
				break;
			}
			else if (std::abs(m_block.getBlockR()[i] - p[i]) < 2*M_EPSILON) {
				norm[i] = FPCONST(1.0);
				break;
			}
		}
		Assert(i < p.dim);

		Float minDiff = M_MAX;
		Float minDir = FPCONST(0.0);
		VectorType<Float> normalt;
		normalt.zero();
		int chosenI = p.dim;
		for (i = 0; i < p.dim; ++i) {
			Float diff = std::abs(m_block.getBlockL()[i] - p[i]);
			if (diff < minDiff) {
				minDiff = diff;
				chosenI = i;
				minDir = -FPCONST(1.0);
			}
			diff = std::abs(m_block.getBlockR()[i] - p[i]);
			if (diff < minDiff) {
				minDiff = diff;
				chosenI = i;
				minDir = FPCONST(1.0);
			}
		}
		normalt[chosenI] = minDir;
		Assert(normalt == norm);
		norm = normalt; // A HACK

		// check if we hit the sensor plane
		if( std::abs(m_camera.getDir().x - norm.x) < M_EPSILON &&
				std::abs(m_camera.getDir().y - norm.y) < M_EPSILON &&
				std::abs(m_camera.getDir().z - norm.z) < M_EPSILON)
			return true;

		// if not, routine
        m_bsdf.sample(d, norm, sampler, d1);
		if (tvec::dot(d1, norm) < FPCONST(0.0)) {
			// re-enter the medium through reflection
			d = d1;
		} else {
			return false;
		}

    	traceTillBlock(p, d, LargeDist, disx, disy, totalOpticalDistance, scaling);
    	distToSensor += disy;
    	LargeDist -= disy;
	}

	return true;
}


template <template <typename> class VectorType>
bool Scene<VectorType>::movePhoton(VectorType<Float> &p, VectorType<Float> &d,
									Float dist, Float &totalOpticalDistance, smp::Sampler &sampler, const Float &scaling) const {

	// Algorithm
	// 1. Move till you reach the boundary or till the distance is reached.
	// 2. If you reached the boundary, reflect with probability and keep progressing TODO: change to weight


	Float disx, disy;
	VectorType<Float> d1, norm;
	traceTillBlock(p, d, dist, disx, disy, totalOpticalDistance, scaling);

	dist -= static_cast<Float>(disy);

	while(dist > M_EPSILON){
		int i;
		norm.zero();
		for (i = 0; i < p.dim; ++i) {
			if (std::abs(m_block.getBlockL()[i] - p[i]) < M_EPSILON) {
				norm[i] = -FPCONST(1.0);
				break;
			}
			else if (std::abs(m_block.getBlockR()[i] - p[i]) < M_EPSILON) {
				norm[i] = FPCONST(1.0);
				break;
			}
		}
		Assert(i < p.dim);

		Float minDiff = M_MAX;
		Float minDir = FPCONST(0.0);
		VectorType<Float> normalt;
		normalt.zero();
		int chosenI = p.dim;
		for (i = 0; i < p.dim; ++i) {
			Float diff = std::abs(m_block.getBlockL()[i] - p[i]);
			if (diff < minDiff) {
				minDiff = diff;
				chosenI = i;
				minDir = -FPCONST(1.0);
			}
			diff = std::abs(m_block.getBlockR()[i] - p[i]);
			if (diff < minDiff) {
				minDiff = diff;
				chosenI = i;
				minDir = FPCONST(1.0);
			}
		}
		normalt[chosenI] = minDir;
		Assert(normalt == norm);
		norm = normalt;

		/*
		 * TODO: I think that, because we always return to same medium (we ignore
		 * refraction), there is no need to adjust radiance by eta*eta.
		 */
		Float magnitude = d.length();
#ifdef PRINT_DEBUGLOG
		std::cout << "Before BSDF sample, d: (" << d.x/magnitude << ", " << d.y/magnitude <<  ", " << d.z/magnitude << "); \n "
				"norm: (" << norm.x << ", " << norm.y << ", " << norm.z << ");" << "A Sampler: " << sampler() << "\n";
#endif
        m_bsdf.sample(d/magnitude, norm, sampler, d1);
        if (tvec::dot(d1, norm) < FPCONST(0.0)) {
			// re-enter the medium through reflection
			d = d1*magnitude;
		} else {
			return false;
		}

    	traceTillBlock(p, d, dist, disx, disy, totalOpticalDistance, scaling);
    	dist -= static_cast<Float>(disy);
	}
	return true;
}

template <>
void Scene<tvec::TVector3>::addEnergyToImage(image::SmallImage &img, const tvec::Vec3f &p,
							Float pathlength, Float val) const {

	Float x = tvec::dot(m_camera.getHorizontal(), p) - m_camera.getOrigin().y;
	Float y = tvec::dot(m_camera.getVertical(), p) - m_camera.getOrigin().z;

	Assert(((std::abs(x) < FPCONST(0.5) * m_camera.getPlane().x)
				&& (std::abs(y) < FPCONST(0.5) * m_camera.getPlane().y)));
	if (((m_camera.getPathlengthRange().x == -1) && (m_camera.getPathlengthRange().y == -1)) ||
		((pathlength > m_camera.getPathlengthRange().x) && (pathlength < m_camera.getPathlengthRange().y))) {
		x = (x / m_camera.getPlane().x + FPCONST(0.5)) * static_cast<Float>(img.getXRes());
		y = (y / m_camera.getPlane().y + FPCONST(0.5)) * static_cast<Float>(img.getYRes());

//		int ix = static_cast<int>(img.getXRes()/2) + static_cast<int>(std::floor(x));
//		int iy = static_cast<int>(img.getYRes()/2) + static_cast<int>(std::floor(y));
		int ix = static_cast<int>(std::floor(x));
		int iy = static_cast<int>(std::floor(y));

		int iz;
		if ((m_camera.getPathlengthRange().x == -1) && (m_camera.getPathlengthRange().y == -1)) {
			iz = 0;
		} else {
			Float z = pathlength - m_camera.getPathlengthRange().x;
			Float range = m_camera.getPathlengthRange().y - m_camera.getPathlengthRange().x;
			z = (z / range) * static_cast<Float>(img.getZRes());
			iz = static_cast<int>(std::floor(z));
		}
#ifdef USE_PIXEL_SHARING
		Float fx = x - std::floor(x);
		Float fy = y - std::floor(y);

		addPixel(img, ix, iy, iz, val*(FPCONST(1.0) - fx)*(FPCONST(1.0) - fy));
		addPixel(img, ix + 1, iy, iz, val*fx*(FPCONST(1.0) - fy));
		addPixel(img, ix, iy + 1, iz, val*(FPCONST(1.0) - fx)*fy);
		addPixel(img, ix + 1, iy + 1, iz, val*fx*fy);
#else
		addPixel(img, ix, iy, iz, val);
#endif
    }
}

template <>
void Scene<tvec::TVector2>::addEnergyToImage(image::SmallImage &img, const tvec::Vec2f &p,
							Float pathlength, Float val) const {

	Float x = tvec::dot(m_camera.getHorizontal(), p) - m_camera.getOrigin().y;

	Assert(std::abs(x) < FPCONST(0.5) * m_camera.getPlane().x);
	if (((m_camera.getPathlengthRange().x == -1) && (m_camera.getPathlengthRange().y == -1)) ||
		((pathlength > m_camera.getPathlengthRange().x) && (pathlength < m_camera.getPathlengthRange().y))) {
		x = (x / m_camera.getPlane().x + FPCONST(0.5)) * static_cast<Float>(img.getXRes());

//		int ix = static_cast<int>(img.getXRes()/2) + static_cast<int>(std::floor(x));
//		int iy = static_cast<int>(img.getYRes()/2) + static_cast<int>(std::floor(y));
		int ix = static_cast<int>(std::floor(x));

		int iz;
		if ((m_camera.getPathlengthRange().x == -1) && (m_camera.getPathlengthRange().y == -1)) {
			iz = 0;
		} else {
			Float z = pathlength - m_camera.getPathlengthRange().x;
			Float range = m_camera.getPathlengthRange().y - m_camera.getPathlengthRange().x;
			z = (z / range) * static_cast<Float>(img.getZRes());
			iz = static_cast<int>(std::floor(z));
		}
#ifdef USE_PIXEL_SHARING
		Float fx = x - std::floor(x);

		addPixel(img, ix, 0, iz, val*(FPCONST(1.0) - fx));
		addPixel(img, ix + 1, 0, iz, val*fx);
#else
		addPixel(img, ix, 0, iz, val);
#endif
    }
}

template <template <typename> class VectorType>
void Scene<VectorType>::addEnergyInParticle(image::SmallImage &img,
			const VectorType<Float> &p, const VectorType<Float> &d, Float distTravelled,
			Float val, const med::Medium &medium, smp::Sampler &sampler, const Float &scaling) const {

	VectorType<Float> p1 = p;

	VectorType<Float> dirToSensor;

	sampleRandomDirection(dirToSensor, sampler); // Samples by assuming that the sensor is in +x direction.

//	if(m_camera.getOrigin().x < m_source.getOrigin().x) // Direction to sensor is flipped. Compensate
//		dirToSensor.x = -dirToSensor.x;

#ifdef PRINT_DEBUGLOG
	std::cout << "dirToSensor: (" << dirToSensor.x << ", " << dirToSensor.y << ", " << dirToSensor.z << ") \n";
#endif


#ifndef OMEGA_TRACKING
	dirToSensor *= getMediumIor(p1, scaling);
#endif

	Float distToSensor;
	if(!movePhotonTillSensor(p1, dirToSensor, distToSensor, distTravelled, sampler, scaling))
		return;

//#ifdef OMEGA_TRACKING
	dirToSensor.normalize();
//#endif

	VectorType<Float> refrDirToSensor = dirToSensor;
	Float fresnelWeight = FPCONST(1.0);
	Float ior = getMediumIor(p1, scaling);

	if (ior > FPCONST(1.0)) {
		refrDirToSensor.x = refrDirToSensor.x/ior;
		refrDirToSensor.normalize(); 
#ifdef PRINT_DEBUGLOG
        std::cout << "refrDir: (" << refrDirToSensor[0] << ", " <<  refrDirToSensor[1] << ", " << refrDirToSensor[2] << ");" << std::endl;
#endif
#ifndef USE_NO_FRESNEL
		fresnelWeight = (FPCONST(1.0) -
		util::fresnelDielectric(dirToSensor.x, refrDirToSensor.x,
			FPCONST(1.0) / ior))
			/ ior / ior;
#endif
	}
	Float foreshortening = dot(refrDirToSensor, m_camera.getDir())/dot(dirToSensor, m_camera.getDir());
	Assert(foreshortening >= FPCONST(0.0));

#if USE_SIMPLIFIED_TIMING
	Float totalOpticalDistance = (distTravelled + distToSensor) * m_ior;
#else
	Float totalOpticalDistance = distTravelled;
#endif

	Float distanceToSensor = 0;
	if(!m_camera.propagateTillSensor(p1, refrDirToSensor, distanceToSensor))
		return;
	totalOpticalDistance += distanceToSensor;

	Float totalPhotonValue = val*(2*M_PI)
			* std::exp(-medium.getSigmaT() * distToSensor)
			* medium.getPhaseFunction()->f(d/d.length(), dirToSensor) // FIXME: Should be refractive index
			* foreshortening
			* fresnelWeight;
	addEnergyToImage(img, p1, totalOpticalDistance, totalPhotonValue);
#ifdef PRINT_DEBUGLOG
    std::cout << "Added Energy:" << totalPhotonValue << " to (" << p1.x << ", " << p1.y << ", " << p1.z << ") at time:" << totalOpticalDistance << std::endl;
    std::cout << "val term:" << val << std::endl;
    std::cout << "exp term:" << std::exp(-medium.getSigmaT() * distToSensor) << std::endl;
    std::cout << "phase function term:" << medium.getPhaseFunction()->f(d/d.length(), dirToSensor) << std::endl;
    std::cout << "fresnel weight:" << fresnelWeight << std::endl;
#endif
}

template <template <typename> class VectorType>
void Scene<VectorType>::addEnergy(image::SmallImage &img,
			const VectorType<Float> &p, const VectorType<Float> &d, Float distTravelled,
			Float val, const med::Medium &medium, smp::Sampler &sampler, const Float& scaling,
			scn::NEECostFunction<VectorType> &costFunction, Problem &problem, Float *initialization) const {

#ifdef USE_WEIGHT_NORMALIZATION
	val *=	static_cast<Float>(img.getXRes()) * static_cast<Float>(img.getYRes())
		/ (m_camera.getPlane().x * m_camera.getPlane().y);
#ifdef USE_PRINTING
		std::cout << "using weight normalization " << std::endl;
#endif
#endif

	VectorType<Float> sensorPoint;
	if(m_camera.samplePosition(sensorPoint, sampler)) {
		/*
		 * TODO: Not sure why this check is here, but it was in the original code.
		 */
		Assert(m_block.inside(sensorPoint));

		// make a direct connection here and accurately measure the radiance

		Float weight = (Float) 1.0;
		VectorType<Float> dirToSensor;
		Float distToSensor;

#ifdef RUNTIME_DEBUGLOG
		auto start = std::chrono::high_resolution_clock::now();
		std::ios_base::sync_with_stdio(false);
#endif
	    bool b = makeSurfaceDirectConnection(p, sensorPoint, scaling, sampler, distTravelled, dirToSensor, distToSensor, weight, costFunction, problem, initialization);
#ifdef RUNTIME_DEBUGLOG
	    auto end = std::chrono::high_resolution_clock::now();
	    double time_taken =
	    		std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
	    if(time_taken > 1e8){
	    	std::cout << "Direct connection took:" <<    time_taken * 1e-9 << " sec" << std::endl;
	    	std::cout << "Status: " << (b ?"Success":"Failed") << std::endl;
			std::cout << "While trying to connect: " << std::endl;
			std::cout << "P1: (" << p.x  << ", " << p.y  << ", " << p.z  << "); " << std::endl;
			std::cout << "P2: (" << sensorPoint.x  << ", " << sensorPoint.y  << ", " << sensorPoint.z  << "); " << std::endl;
	    }
#endif

		if(!b)
			return;

		VectorType<Float> refrDirToSensor = dirToSensor;
		Float fresnelWeight = FPCONST(1.0);

		if (m_ior > FPCONST(1.0)) {
			for (int iter = 1; iter < dirToSensor.dim; ++iter) {
				refrDirToSensor[iter] = dirToSensor[iter] * m_ior;
			}
			refrDirToSensor.normalize();
#ifdef PRINT_DEBUGLOG
        std::cout << "refrDir: (" << refrDirToSensor[0] << ", " <<  refrDirToSensor[1] << ", " << refrDirToSensor[2] << ");" << std::endl;
#endif
#ifndef USE_NO_FRESNEL
			fresnelWeight = (FPCONST(1.0) -
			util::fresnelDielectric(dirToSensor.x, refrDirToSensor.x,
				FPCONST(1.0) / m_ior))
				/ m_ior / m_ior;
#endif
		}

		/*
		 * TODO: Double-check that the foreshortening term is needed, and
		 * that it is applied after refraction.
		 */
		Float foreshortening = dot(refrDirToSensor, m_camera.getDir());
		Assert(foreshortening >= FPCONST(0.0));

#if USE_SIMPLIFIED_TIMING
    Float totalOpticalDistance = (distTravelled + distToSensor) * m_ior;
#else
    Float totalOpticalDistance = distTravelled;
#endif

		Float falloff = FPCONST(1.0);
		if (p.dim == 2) {
			falloff = distToSensor;
		} else if (p.dim == 3) {
			falloff = distToSensor * distToSensor;
		}

		Float totalPhotonValue = val * m_camera.getPlane().x * m_camera.getPlane().y
				* std::exp(- medium.getSigmaT() * distToSensor)
				* medium.getPhaseFunction()->f(d/d.length(), dirToSensor)
				* fresnelWeight
				* weight
				* foreshortening
				/ falloff;
		addEnergyToImage(img, sensorPoint, totalOpticalDistance, totalPhotonValue);
#ifdef PRINT_DEBUGLOG
    std::cout << "Added Energy:" << totalPhotonValue << " to (" << sensorPoint.x << ", " << sensorPoint.y << ", " << sensorPoint.z << ") at time:" << totalOpticalDistance << std::endl;
    std::cout << "val term:" << val << std::endl;
    std::cout << "exp term:" << std::exp(-medium.getSigmaT() * distToSensor) << std::endl;
    std::cout << "phase function term:" << medium.getPhaseFunction()->f(d/d.length(), dirToSensor) << std::endl;
    std::cout << "fresnel weight:" << fresnelWeight << std::endl;
//    std::cout << "weight:" << weight << std::endl;
    std::cout << "foreshortening:" << foreshortening << std::endl;
    std::cout << "falloff:" << falloff << std::endl;
#endif

#ifdef RUNTIME_DEBUGLOG
    if(totalPhotonValue > 1e2){
		std::cout << "Added Energy:" << totalPhotonValue << " to (" << sensorPoint.x << ", " << sensorPoint.y << ", " << sensorPoint.z << ") at time:" << totalOpticalDistance << std::endl;
		std::cout << "val term:" << val << std::endl;
		std::cout << "exp term:" << std::exp(-medium.getSigmaT() * distToSensor) << std::endl;
		std::cout << "phase function term:" << medium.getPhaseFunction()->f(d/d.length(), dirToSensor) << std::endl;
		std::cout << "fresnel weight:" << fresnelWeight << std::endl;
		std::cout << "weight:" << weight << std::endl;
		std::cout << "foreshortening:" << foreshortening << std::endl;
		std::cout << "falloff:" << falloff << std::endl;
	}
#endif

	}
}

template <template <typename> class VectorType>
void Scene<VectorType>::addEnergyDeriv(image::SmallImage &img, image::SmallImage &dSigmaT,
						image::SmallImage &dAlbedo, image::SmallImage &dGVal,
						const VectorType<Float> &p, const VectorType<Float> &d,
						Float distTravelled, Float val, Float sumScoreSigmaT,
						Float sumScoreAlbedo, Float sumScoreGVal,
						const med::Medium &medium, smp::Sampler &sampler) const {

#ifdef USE_WEIGHT_NORMALIZATION
	val *=	static_cast<Float>(img.getXRes()) * static_cast<Float>(img.getYRes())
		/ (m_camera.getPlane().x * m_camera.getPlane().y);
#ifdef USE_PRINTING
		std::cout << "using weight normalization " << std::endl;
#endif
#endif
#ifdef USE_PRINTING
	std::cout << "total = " << distTravelled << std::endl;
#endif

	VectorType<Float> sensorPoint;
	if(m_camera.samplePosition(sensorPoint, sampler)) {
		/*
		 * TODO: Not sure why this check is here, but it was in the original code.
		 */
		Assert(m_block.inside(sensorPoint));

		VectorType<Float> distVec = sensorPoint - p;
		Float distToSensor = distVec.length();

		VectorType<Float> dirToSensor = distVec;
		dirToSensor.normalize();

		VectorType<Float> refrDirToSensor = dirToSensor;
		Float fresnelWeight = FPCONST(1.0);

		if (m_ior > FPCONST(1.0)) {
			Float sqrSum = FPCONST(0.0);
			for (int iter = 1; iter < dirToSensor.dim; ++iter) {
				refrDirToSensor[iter] = dirToSensor[iter] * m_ior;
				sqrSum += refrDirToSensor[iter] * refrDirToSensor[iter];
			}
			refrDirToSensor.x = std::sqrt(FPCONST(1.0) - sqrSum);
			if (dirToSensor.x < FPCONST(0.0)) {
				refrDirToSensor.x *= -FPCONST(1.0);
			}
#ifndef USE_NO_FRESNEL
			fresnelWeight = (FPCONST(1.0) -
			util::fresnelDielectric(dirToSensor.x, refrDirToSensor.x,
				FPCONST(1.0) / m_ior))
				/ m_ior / m_ior;
#endif
		}

		/*
		 * TODO: Double-check that the foreshortening term is needed, and that
		 * it is applied after refraction.
		 */
		Float foreshortening = dot(refrDirToSensor, m_camera.getDir());
		Assert(foreshortening >= FPCONST(0.0));

		Float totalDistance = (distTravelled + distToSensor) * m_ior;
		Float falloff = FPCONST(1.0);
		if (p.dim == 2) {
			falloff = distToSensor;
		} else if (p.dim == 3) {
			falloff = distToSensor * distToSensor;
		}
		Float totalPhotonValue = val
				* std::exp(- medium.getSigmaT() * distToSensor)
				* medium.getPhaseFunction()->f(d, dirToSensor)
				* fresnelWeight
				* foreshortening
				/ falloff;
		addEnergyToImage(img, sensorPoint, totalDistance, totalPhotonValue);
		Float valDSigmaT = totalPhotonValue * (sumScoreSigmaT - distToSensor);
		addEnergyToImage(dSigmaT, sensorPoint, totalDistance, valDSigmaT);
		Float valDAlbedo = totalPhotonValue * sumScoreAlbedo;
		addEnergyToImage(dAlbedo, sensorPoint, totalDistance, valDAlbedo);
		Float valDGVal = totalPhotonValue *
				(sumScoreGVal + medium.getPhaseFunction()->score(d, dirToSensor));
		addEnergyToImage(dGVal, sensorPoint, totalDistance, valDGVal);
	}
}

//template class Block<tvec::TVector2>;
template class Block<tvec::TVector3>;
//template class Camera<tvec::TVector2>;
template class Camera<tvec::TVector3>;
//template class AreaSource<tvec::TVector2>;
template class AreaSource<tvec::TVector3>;
//template class AreaTexturedSource<tvec::TVector2>;
template class AreaTexturedSource<tvec::TVector3>;
//template class US<tvec::TVector2>;
template class US<tvec::TVector3>;
//template class Scene<tvec::TVector2>;
template class Scene<tvec::TVector3>;
//template class NEECostFunction<tvec::TVector2>;
template class NEECostFunction<tvec::TVector3>;


}	/* namespace scn */
