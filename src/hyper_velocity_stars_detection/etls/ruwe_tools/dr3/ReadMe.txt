Gaia (E)DR3: Re-normalised Unit Weight Error (RUWE) - tables of u0(g,c)

L. Lindegren (2023 Sep 13)


The file table_u0_g_c_p5.txt is a lookup table for the function u0(g,c) used
to compute RUWE for five-parameter solutions (astrometric_params_solved = 31)
according to

RUWE = UWE / u0(G,C).

Here,

UWE = sqrt(astrometric_chi2_al/(astrometric_n_good_obs_al-N)),
N = 5,
g = phot_g_mean_mag, and
c = nu_eff_used_in_astrometry.

Similarly, the file table_u0_g_c_p6.txt contains u0(g,c) for six-parameter solutions
(astrometric_params_solved = 95), with notations as above except that N = 6
and c = pseudocolour.


Detailed file descriptions:

The files contain comma separated values (CSV) including a single header line
with unique column names (g, c, u0). The files can be read e.g. in TOPCAT using
format specification CSV.

Both files have 85951 rows of data (plus one header line), with all combinations of
g = 4.00(0.02)21.00 (851 different values of g) and c = 1.00(0.01)2.00 (101 different
values of c). For other values of g and c within their their tabulated ranges ([4,21]
and [1,2]) linear interpolation should be used. Beyond that, the nearest tabulated
value should be used.


Other information:

The general principles for estimating u0(g,c) are similar to the ones used for DR2
(see GAIA-C3-TN-LU-LL-124, https://dms.cosmos.esa.int/COSMOS/doc_fetch.php?id=3757412).
In particular, u0(g,c) represents the (smoothed) 41st percentile of UWE for stars of
the given magnitude and colour.

The functions are displayed in the attached plot_u0_g_c_p5.pdf and plot_u0_g_c_p6.pdf.
