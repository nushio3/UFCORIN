/*
  g++ demonstrate-wavelet.cpp  -L/usr/local/lib -lgsl -lgslcblas -lm -o demo.out
*/

#include <string>
#include <stdio.h>
#include <math.h>
#include <gsl/gsl_sort.h>
#include <gsl/gsl_wavelet.h>
#include <gsl/gsl_wavelet2d.h>
#include <climits>

using namespace std;

const int n = 64;

void draw_pat(int pat_id, double *data) {
  gsl_wavelet *w;
  gsl_wavelet_workspace *work;

  work = gsl_wavelet_workspace_alloc (n*n);

  srand(1341398);
  for (int j = 0; j < n; j++)  {
    for (int i = 0; i < n; i++)  {
      data[j*n+i] = 0;
    }
  }

  switch(pat_id) {
  case 0:
    for (int j = 0; j < n; j++)  {
      for (int i = 0; i < n; i++)  {
        if ((i-n/2+0.5)*(i-n/2+0.5) + (j-n/2+0.5)*(j-n/2+0.5) < n*n/4-2*n)
          data[j*n+i] = double(rand())/INT_MAX*2-1;
      }
    }
    break;
  case 1:
    w = gsl_wavelet_alloc (gsl_wavelet_haar_centered, 2);
    data[ 6 + n * 35] = 1.0;
    data[12 + n * 40] = 1.0;
    data[24 + n * 45] = 1.0;
    data[48 + n * 50] = 1.0;
    gsl_wavelet2d_transform_inverse (w, data, n, n, n, work);
    gsl_wavelet_free (w);

    break;
  case 2:
    w = gsl_wavelet_alloc (gsl_wavelet_haar_centered, 2);
    data[62 + n * 62] = 1.0;
    data[ 2 + n * 30] = 1.0;
    data[14 + n *  7] = 1.0;
    gsl_wavelet2d_nstransform_inverse (w, data, n, n, n, work);
    gsl_wavelet_free (w);
    break;
  case 3:
    w = gsl_wavelet_alloc (gsl_wavelet_bspline_centered, 301);
    data[62 + n * 62] = 1.0;
    data[ 2 + n * 30] = 1.0;
    data[14 + n *  7] = 1.0;
    gsl_wavelet2d_nstransform_inverse (w, data, n, n, n, work);
    gsl_wavelet_free (w);
    break;

  }

  gsl_wavelet_workspace_free (work);

}

int
main (int argc, char **argv) {
  int i,j;
  double *data = (double*)malloc (n * n * sizeof (double));
  double *abscoeff = (double*)malloc (n * n * sizeof (double));
  size_t *p = (size_t*)malloc (n * n * n* sizeof (size_t));

  FILE * fp;
  gsl_wavelet *w;
  gsl_wavelet_workspace *work;


  for(int pat_id=0; pat_id < 4; ++pat_id){
    work = gsl_wavelet_workspace_alloc (n*n);

    draw_pat(pat_id,data);

    string header = "demosun";
    if (pat_id==1) header = "tp-S";
    if (pat_id==2) header = "tp-N";
    if (pat_id==3) header = "bs-N";

    if (pat_id==3) {
      w = gsl_wavelet_alloc (gsl_wavelet_bspline_centered, 301);
    } else {
      w = gsl_wavelet_alloc (gsl_wavelet_haar_centered, 2);
    }


    fp = fopen((header + "-real.txt").c_str(),"w");
    for (j = 0; j < n; j++)  {
      for (int iter=0;iter<2;++iter) {
        double yoff = iter==0 ? 0.001 : 0.999;
        for (i = 0; i < n; i++)  {
          fprintf(fp, "%f %f %f\n", i+0.001,j+yoff,data[j*n+i]);
          fprintf(fp, "%f %f %f\n", i+0.999,j+yoff,data[j*n+i]);
        }
        fprintf(fp, "\n");
      }
    }
    fclose(fp);

    gsl_wavelet2d_transform_forward (w, data, n, n, n, work);

    fp = fopen((header + "-WS.txt").c_str(),"w");
    for (j = 0; j < n; j++)  {
      for (int iter=0;iter<2;++iter) {
        double yoff = iter==0 ? 0.001 : 0.999;
        for (i = 0; i < n; i++)  {
          fprintf(fp, "%f %f %f\n", i+0.001,j+yoff,data[j*n+i]);
          fprintf(fp, "%f %f %f\n", i+0.999,j+yoff,data[j*n+i]);
        }
        fprintf(fp, "\n");
      }
    }
    fclose(fp);

    draw_pat(pat_id,data);

    gsl_wavelet2d_nstransform_forward (w, data, n, n, n, work);

    fp = fopen((header + "-WN.txt").c_str(),"w");
    for (j = 0; j < n; j++)  {
      for (int iter=0;iter<2;++iter) {
        double yoff = iter==0 ? 0.001 : 0.999;
        for (i = 0; i < n; i++)  {
          fprintf(fp, "%f %f %f\n", i+0.001,j+yoff,data[j*n+i]);
          fprintf(fp, "%f %f %f\n", i+0.999,j+yoff,data[j*n+i]);
        }
        fprintf(fp, "\n");
      }
    }
    fclose(fp);


    gsl_wavelet_free (w);
    gsl_wavelet_workspace_free (work);
  }
  free (data);
  free (abscoeff);
  free (p);
  return 0;
}
