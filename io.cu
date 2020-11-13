#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "harris.h"

typedef unsigned char Pixel;
#define PGM_MAX 255
#define PPM_TYPE 1
#define PGM_TYPE 2



void error(char *message)
{
	fprintf(stderr," %s\n", message);
	exit(1);
}

/************************************************************************
 * Checks for white spaces.
 *************************************************************************/
int isWhiteSpace(int num)
{
	if ((num == ' ') || (num == '\t') || (num == '\n') || (num == '\r'))
		return 1;
	else
		return 0;
}

/************************************************************************
 * Converts a string to an int ignoring white spaces and comment lines.
 *************************************************************************/

int char2Int(FILE *f)
{
	int num;
	char ch;

	num = 0;
	do
	{
		do
		{
			fread(&ch, sizeof(char), 1, f);
		} while (isWhiteSpace(ch) == 1);
		if (ch == '#')
			while (ch != '\n')
				fread(&ch, sizeof(char), 1, f);
	} while (isWhiteSpace(ch) == 1);
	while (('0' <= ch) && (ch <= '9'))
	{
		num = 10*num + ch - '0';
		fread(&ch, sizeof(char), 1, f);
	}
	return num;
}

/************************************************************************
 * Reads the header of a PGM file, returning the image dimensions. The file name
 * is in name. As output you get the image dimensions in rows and columns.
 ***********************************************************************/
void pgmReadHeader(char *name, int *rows, int *columns)
{
	FILE *fp;
	char P, type, buf[256];

	fp = fopen(name, "rb");
	if (fp == NULL){
		sprintf(buf, "Open error for file %s", name);
		error(buf);
	}

	fread(&P, sizeof(char), 1, fp);
	fread(&type, sizeof(char), 1, fp);
	if ((P != 'P') || (type != '5')){
		fprintf(stderr, "%s is not  a raw PGM file\n", name);
		exit(1);
	}

	*columns = char2Int(fp);
	*rows = char2Int(fp);
	fclose(fp);
}

/************************************************************************
 * Reads the header of a PPM file, returning the image dimensions. The file name
 * is in name. As output you get the image dimensions in rows and columns.
 ***********************************************************************/
void ppmReadHeader(char *name, int *rows, int *columns)
{
	FILE *fp;
	char P, type, buf[256];

	fp = fopen(name, "rb");
	if (fp == NULL){
		sprintf(buf, "Open error for file %s", name);
		error(buf);
	}

	fread(&P, sizeof(char), 1, fp);
	fread(&type, sizeof(char), 1, fp);
	if ((P != 'P') || (type != '6')){
		fprintf(stderr, "%s is not a raw PPM file\n", name);
		exit(1);
	}

	*columns = char2Int(fp);
	*rows = char2Int(fp);
	fclose(fp);
}

/*************************************************************************
 * Converts an int to a string and writes it to a file.
 *************************************************************************/
void int2Char(int num, FILE *f)
{
	int count;
	char m[40];

	count = 0;
	do
	{
		m[count] = num%10 + '0';
		num -= num % 10;
		num /= 10;
		count++;
	} while (num > 0);

	for (count = count - 1; count >= 0; count--)
		fwrite(&m[count], sizeof(char), 1, f);
}

/************************************************************************
 * Reads a PGM file format. The file name is in name. As output you get
 * the image dimensions in rows and columns while an array is filled
 * with the image values.
 ***********************************************************************/
int pgmReadBuffer(char *name, Pixel *img, int *rows, int *columns)
{
	FILE *fp;
	char P, type, buf[256];

	fp = fopen(name, "rb");
	if (fp == NULL){
		sprintf(buf, "Open error for file %s", name);
		error(buf);
	}

	fread(&P, sizeof(char), 1, fp);
	fread(&type, sizeof(char), 1, fp);
	if ((P != 'P') || (type != '5')){
		/*fprintf(stderr, "%s is not  a raw PGM file\n", name);*/
		return 0;
	}

	*columns = char2Int(fp);
	*rows = char2Int(fp);

	fread(img, sizeof(Pixel), (*rows)*(*columns), fp);
	fclose(fp);

	return 1;
}

/*************************************************************************
 * Reads a PPM file format. The file name is in name. As output you get
 * the image dimensions in rows and columns while a 3D array is filled
 * with the image values in RGB.
 *************************************************************************/
int ppmReadBuffer(char *name, Pixel *img, int *rows, int *columns)
{
	FILE *fp;
	char P, type, buf[256];

	fp = fopen(name, "rb");
	if (fp == NULL){
		sprintf(buf, "Open error for file %s", name);
		error(buf);
	}

	fread(&P, sizeof(char), 1, fp);
	fread(&type, sizeof(char), 1, fp);
	if ((P != 'P') || (type != '6')){
		/*fprintf(stderr, "%s is not a raw PPM file\n", name);*/
		return 0;
	}

	*columns = char2Int(fp);
	*rows = char2Int(fp);

	fread(img, sizeof(Pixel), 3*(*rows)*(*columns), fp);
	fclose(fp);

	return 1;
}

/*************************************************************************
 * Reads either a PPM or a PGM image
 *************************************************************************/
int ppmOrpgmReadBuffer(char *name, Pixel *img, int *rows, int *columns)
{
	int res;

	res=ppmReadBuffer(name, img, rows, columns);
	if(res) return PPM_TYPE;

	res=pgmReadBuffer(name, img, rows, columns);
	if(res) return PGM_TYPE;

	fprintf(stderr, "%s is not a raw PPM or PGM file\n", name);
	exit(1);

	return 0; /* will never get here */
}

/*************************************************************************
 * Writes the image having dimensions (rows x columns) to a file in PGM
 * format. PGM format is described in prgRead function comments.
 *************************************************************************/
void pgmWriteBuffer(Pixel *img, int rows, int columns, char *name)
{
	char buf[256];
	FILE *f;

	f = fopen(name, "wb");
	if (f == NULL){
		sprintf(buf, "Write error for file %s", name);
		error(buf);
	}

	fwrite("P5\n", sizeof(char), 3, f);
	fwrite("# FREEWARE\n", sizeof(char),11,f);
	int2Char(columns, f);
	fwrite(" ", sizeof(char), 1, f);
	int2Char(rows, f);
	fwrite("\n", sizeof(char), 1, f);
	int2Char(PGM_MAX, f);
	fwrite("\n", sizeof(char), 1, f);

	fwrite(img, sizeof(Pixel), rows*columns, f);

	fclose(f);
}

/*************************************************************************
 * Writes the color image having dimensions (rows x columns x 3) to a file
 * in PPM format. PPM format is described in ppmRead function comments.
 *************************************************************************/
void ppmWriteBuffer(Pixel *img, int rows, int columns, char *name)
{
	char buf[256];
	FILE *f;

	f = fopen(name, "wb");
	if (f == NULL){
		sprintf(buf, "Write error for file %s", name);
		error(buf);
	}

	fwrite("P6\n", sizeof(char), 3, f);
	fwrite("# FREEWARE\n", sizeof(char),11,f);
	int2Char(columns, f);
	fwrite(" ", sizeof(char), 1, f);
	int2Char(rows, f);
	fwrite("\n", sizeof(char), 1, f);
	int2Char(PGM_MAX, f);
	fwrite("\n", sizeof(char), 1, f);

	fwrite(img, sizeof(Pixel), rows*columns*3, f);

	fclose(f);
}