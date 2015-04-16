/*
  g++ demonstrate-wavelet.cpp  -L/usr/local/lib -lgsl -lgslcblas -lm -o demo.out
*/


#include <stdio.h>
#include <math.h>
#include <gsl/gsl_sort.h>
#include <gsl/gsl_wavelet.h>
#include <gsl/gsl_wavelet2d.h>
#include <climits>

int
main (int argc, char **argv) {
  int i,j, n = 256, nc = 20;
  double *data = (double*)malloc (n * n * sizeof (double));
  double *abscoeff = (double*)malloc (n * n * sizeof (double));
  size_t *p = (size_t*)malloc (n * n * n* sizeof (size_t));

  FILE * f;
  gsl_wavelet *w;
  gsl_wavelet_workspace *work;

  w = gsl_wavelet_alloc (gsl_wavelet_haar_centered, 2);
  work = gsl_wavelet_workspace_alloc (n*n);

  for (j = 0; j < n; j++)  {
    for (i = 0; i < n; i++)  {
      data[j*n+i] = 0;
      if ((i-n/2)*(i-n/2) + (j-n/2)*(j-n/2) < n*n/4)
        data[j*n+i] = double(rand())/INT_MAX;
    }
  }
  //data[7*n+7] = 1;

  gsl_wavelet2d_transform_forward (w, data, n, n, n, work);


  for (j = 0; j < n; j++)  {
    for (i = 0; i < n; i++)  {
      printf ("%d %d %f\n", i,j,data[j*n+i]);
    }
    printf ("\n");
  }

  gsl_wavelet_free (w);
  gsl_wavelet_workspace_free (work);

  free (data);
  free (abscoeff);
  free (p);
  return 0;
}
