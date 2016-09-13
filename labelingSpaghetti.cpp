// Copyright(c) 2016 - Costantino Grana, Federico Bolelli, Lorenzo Baraldi and Roberto Vezzani
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met :
// 
// *Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// 
// * Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and / or other materials provided with the distribution.
// 
// * Neither the name of YACCLAB nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED.IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "labelingSpaghetti.h"

using namespace cv;
using namespace std;

//Find the root of the tree of node i
//template<typename LabelT>
inline static
uint findRoot(const uint *P, uint i){
	uint root = i;
	while (P[root] < root){
		root = P[root];
	}
	return root;
}

//Make all nodes in the path of node i point to root
//template<typename LabelT>
inline static
void setRoot(uint *P, uint i, uint root){
	while (P[i] < i){
		uint j = P[i];
		P[i] = root;
		i = j;
	}
	P[i] = root;
}

//Find the root of the tree of the node i and compress the path in the process
//template<typename LabelT>
inline static
uint find(uint *P, uint i){
	uint root = findRoot(P, i);
	setRoot(P, i, root);
	return root;
}

//unite the two trees containing nodes i and j and return the new root
//template<typename LabelT>
inline static
uint set_union(uint *P, uint i, uint j){
	uint root = findRoot(P, i);
	if (i != j){
		uint rootj = findRoot(P, j);
		if (root > rootj){
			root = rootj;
		}
		setRoot(P, j, root);
	}
	setRoot(P, i, root);
	return root;
}

//Flatten the Union Find tree and relabel the components
//template<typename LabelT>
inline static
uint flattenL(uint *P, uint length){
	uint k = 1;
	for (uint i = 1; i < length; ++i){
		if (P[i] < i){
			P[i] = P[P[i]];
		}
		else{
			P[i] = k; k = k + 1;
		}
	}
	return k;
}



#define condition_b imgLabels_row_prev_prev[c-2] & 0x40000000
#define condition_g imgLabels_row_prev_prev[c-2] & 0x20000000
#define condition_h imgLabels_row_prev_prev[c-2] & 0x10000000

#define condition_c imgLabels_row_prev_prev[c] & 0x80000000
#define condition_d imgLabels_row_prev_prev[c] & 0x40000000
#define condition_i imgLabels_row_prev_prev[c] & 0x20000000
#define condition_j imgLabels_row_prev_prev[c] & 0x10000000

#define condition_e imgLabels_row_prev_prev[c+2] & 0x80000000
#define condition_k imgLabels_row_prev_prev[c+2] & 0x20000000

#define condition_m old_pix & 0x80000000
#define condition_n old_pix & 0x40000000
#define condition_r old_pix & 0x10000000

#define condition_o pix & 0x80000000
#define condition_p pix & 0x40000000
#define condition_s pix & 0x20000000
#define condition_t pix & 0x10000000


#define action_1	imgLabels_row[c] = pix | 0; //Action_1: No action (the block has no foreground pixels) 
#define action_2	imgLabels_row[c] = pix | lunique; /*Action_2: New label (the block has foreground pixels and is not connected to anything else) */ \
					P[lunique] = lunique;		\
					lunique = lunique + 1;			
#define action_3	imgLabels_row[c] = pix | (0x0fffffff & imgLabels_row_prev_prev[c - 2]); //Action_3: Assign label of block P
#define action_4	imgLabels_row[c] = pix | (0x0fffffff & imgLabels_row_prev_prev[c]); //Action_4: Assign label of block Q 
#define action_5	imgLabels_row[c] = pix | (0x0fffffff & imgLabels_row_prev_prev[c + 2]);	//Action_5: Assign label of block R
#define action_6	imgLabels_row[c] = pix | (0x0fffffff & imgLabels_row[c - 2]);	//Action_6: Assign label of block S
#define action_7	imgLabels_row[c] = pix | set_union(P, 0x0fffffff & imgLabels_row_prev_prev[c - 2],	0x0fffffff & imgLabels_row_prev_prev[c]);	//Action_7: Merge labels of block P and Q
#define action_8	imgLabels_row[c] = pix | set_union(P, 0x0fffffff & imgLabels_row_prev_prev[c - 2],	0x0fffffff & imgLabels_row_prev_prev[c + 2]);	//Action_8: Merge labels of block P and R
#define action_9	imgLabels_row[c] = pix | set_union(P, 0x0fffffff & imgLabels_row_prev_prev[c - 2],	0x0fffffff & imgLabels_row[c - 2]);	// ACTION_9 Merge labels of block P and S
#define action_10	imgLabels_row[c] = pix | set_union(P, 0x0fffffff & imgLabels_row_prev_prev[c],		0x0fffffff & imgLabels_row_prev_prev[c + 2]);	// ACTION_10 Merge labels of block Q and R
#define action_11	imgLabels_row[c] = pix | set_union(P, 0x0fffffff & imgLabels_row_prev_prev[c],		0x0fffffff & imgLabels_row[c - 2]);	//Action_11: Merge labels of block Q and S
#define action_12	imgLabels_row[c] = pix | set_union(P, 0x0fffffff & imgLabels_row_prev_prev[c + 2],	0x0fffffff & imgLabels_row[c - 2]);	//Action_12: Merge labels of block R and S
//Action_13:	// Merge labels of block P, Q and R
//			imgLabels(r,c) = es.resolve(imgLabels(r-2,c-2),imgLabels(r-2,c),imgLabels(r-2,c+2));
//			continue;
#define action_14	imgLabels_row[c] = pix | set_union(P, set_union(P, 0x0fffffff & imgLabels_row_prev_prev[c - 2], 0x0fffffff & imgLabels_row_prev_prev[c]), 0x0fffffff & imgLabels_row[c - 2]);	//Action_14: Merge labels of block P, Q and S
#define action_15	imgLabels_row[c] = pix | set_union(P, set_union(P, 0x0fffffff & imgLabels_row_prev_prev[c - 2], 0x0fffffff & imgLabels_row_prev_prev[c + 2]), 0x0fffffff & imgLabels_row[c - 2]);	//Action_15: Merge labels of block P, R and S
#define action_16	imgLabels_row[c] = pix | set_union(P, set_union(P, 0x0fffffff & imgLabels_row_prev_prev[c], 0x0fffffff & imgLabels_row_prev_prev[c + 2]), 0x0fffffff & imgLabels_row[c - 2]);	//Action_16: labels of block Q, R and S




inline static
void firstScanSpaghetti(const Mat1b &img, Mat1i& imgLabels, uint* P, uint &lunique) {
	int w(img.cols), h(img.rows);
	int pix = 0, old_pix;

#define read_pixels \
	old_pix = pix; \
	pix  = (img_row[c] > 0) << 31; \
	pix |= (img_row[c + 1] > 0) << 30; \
	pix |= (img_row_fol[c] > 0) << 29; \
	pix |= (img_row_fol[c + 1] > 0) << 28;

#define if_finish_condition	\
	c += 2; \
	read_pixels \
	if (c >= w - 2)

	for (int r = 0; r < 2; r += 2) {
		// Get row pointers
		const uchar* const img_row = img.ptr<uchar>(r);
		const uchar* const img_row_fol = (uchar *)(((char *)img_row) + img.step.p[0]);
		uint* const imgLabels_row = imgLabels.ptr<uint>(r);

		int c = 0;

		read_pixels;

	}

	for (int r = 2; r < h - 1; r += 2) {
		// Get row pointers
		const uchar* const img_row = img.ptr<uchar>(r);
		const uchar* const img_row_fol = (uchar *)(((char *)img_row) + img.step.p[0]);
		uint* const imgLabels_row = imgLabels.ptr<uint>(r);
		uint* const imgLabels_row_prev_prev = (uint *)(((char *)imgLabels_row) - imgLabels.step.p[0] - imgLabels.step.p[0]);

		int c = 0;

		read_pixels;

	}

	for (int r = h - 1; r < h; r += 2) {
		// Get row pointers
		const uchar* const img_row = img.ptr<uchar>(r);
		const uchar* const img_row_fol = (uchar *)(((char *)img_row) + img.step.p[0]);
		uint* const imgLabels_row = imgLabels.ptr<uint>(r);
		uint* const imgLabels_row_prev_prev = (uint *)(((char *)imgLabels_row) - imgLabels.step.p[0] - imgLabels.step.p[0]);

		int c = 0;

		read_pixels;

	}

#undef read_pixels
#undef if_finish_condition
}

inline static
void firstScanSpaghetti_ee(const Mat1b &img, Mat1i& imgLabels, uint* P, uint &lunique) {
	int w(img.cols), h(img.rows);
	int pix = 0, old_pix;

#define fr_finish_condition(n)	\
	c += 2; \
	old_pix = pix; \
	pix  = (img_row[c] > 0) << 31; \
	pix |= (img_row_fol[c] > 0) << 29; \
	pix |= (img_row[c + 1] > 0) << 30; \
	pix |= (img_row_fol[c + 1] > 0) << 28; \
	if (c == w - 2) \
		goto fr_break_##n; 
#define finish_condition(n) \
	c += 2; \
	old_pix = pix; \
	pix = (img_row[c] > 0) << 31; \
	pix |= (img_row_fol[c] > 0) << 29; \
	pix |= (img_row[c + 1] > 0) << 30; \
	pix |= (img_row_fol[c + 1] > 0) << 28; \
	if (c == w - 2) \
		goto break_##n;

	for (int r = 0; r < 2; r += 2) {
		// Get row pointers
		const uchar* const img_row = img.ptr<uchar>(r);
		const uchar* const img_row_fol = (uchar *)(((char *)img_row) + img.step.p[0]);
		uint* const imgLabels_row = imgLabels.ptr<uint>(r);

		int c = -2;
		fr_finish_condition(0);
		if (condition_o) {
			action_2;
			goto fr_tree_9;
		}
		else {
			if (condition_s) {
				if (condition_p) {
					action_2;
					goto fr_tree_1;
				}
				else {
					action_2;
					goto fr_tree_3;
				}
			}
			else {
				if (condition_p) {
					action_2;
					goto fr_tree_1;
				}
				else {
					if (condition_t) {
						action_2;
						goto fr_tree_1;
					}
					else {
						action_1;
						goto fr_tree_0;
					}
				}
			}
		}
	fr_tree_0: fr_finish_condition(0);
		if (condition_o) {
			action_2;
			goto fr_tree_9;
		}
		else {
			if (condition_s) {
				if (condition_p) {
					action_2;
					goto fr_tree_1;
				}
				else {
					action_2;
					goto fr_tree_3;
				}
			}
			else {
				if (condition_p) {
					action_2;
					goto fr_tree_1;
				}
				else {
					if (condition_t) {
						action_2;
						goto fr_tree_1;
					}
					else {
						action_1;
						goto fr_tree_0;
					}
				}
			}
		}
	fr_tree_1: fr_finish_condition(1);
		if (condition_o) {
			action_6;
			goto fr_tree_9;
		}
		else {
			if (condition_s) {
				if (condition_p) {
					action_6;
					goto fr_tree_1;
				}
				else {
					action_6;
					goto fr_tree_3;
				}
			}
			else {
				if (condition_p) {
					action_2;
					goto fr_tree_1;
				}
				else {
					if (condition_t) {
						action_2;
						goto fr_tree_1;
					}
					else {
						action_1;
						goto fr_tree_0;
					}
				}
			}
		}
	fr_tree_3: fr_finish_condition(3);
		if (condition_o) {
			if (condition_r) {
				action_6;
				goto fr_tree_9;
			}
			else {
				action_2;
				goto fr_tree_9;
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_r) {
						action_6;
						goto fr_tree_1;
					}
					else {
						action_2;
						goto fr_tree_1;
					}
				}
				else {
					if (condition_r) {
						action_6;
						goto fr_tree_3;
					}
					else {
						action_2;
						goto fr_tree_3;
					}
				}
			}
			else {
				if (condition_p) {
					action_2;
					goto fr_tree_1;
				}
				else {
					if (condition_t) {
						action_2;
						goto fr_tree_1;
					}
					else {
						action_1;
						goto fr_tree_0;
					}
				}
			}
		}
	fr_tree_9: fr_finish_condition(9);
		if (condition_o) {
			if (condition_n) {
				action_6;
				goto fr_tree_9;
			}
			else {
				if (condition_r) {
					action_6;
					goto fr_tree_9;
				}
				else {
					action_2;
					goto fr_tree_9;
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_n) {
						action_6;
						goto fr_tree_1;
					}
					else {
						if (condition_r) {
							action_6;
							goto fr_tree_1;
						}
						else {
							action_2;
							goto fr_tree_1;
						}
					}
				}
				else {
					if (condition_r) {
						action_6;
						goto fr_tree_3;
					}
					else {
						if (condition_n) {
							action_6;
							goto fr_tree_3;
						}
						else {
							action_2;
							goto fr_tree_3;
						}
					}
				}
			}
			else {
				if (condition_p) {
					action_2;
					goto fr_tree_1;
				}
				else {
					if (condition_t) {
						action_2;
						goto fr_tree_1;
					}
					else {
						action_1;
						goto fr_tree_0;
					}
				}
			}
		}
	fr_break_0:
		if (condition_o) {
			action_2;
		}
		else {
			if (condition_s) {
				if (condition_p) {
					action_2;
				}
				else {
					action_2;
				}
			}
			else {
				if (condition_p) {
					action_2;
				}
				else {
					if (condition_t) {
						action_2;
					}
					else {
						action_1;
					}
				}
			}
		}
		continue;
	fr_break_1:
		if (condition_o) {
			action_6;
		}
		else {
			if (condition_s) {
				if (condition_p) {
					action_6;
				}
				else {
					action_6;
				}
			}
			else {
				if (condition_p) {
					action_2;
				}
				else {
					if (condition_t) {
						action_2;
					}
					else {
						action_1;
					}
				}
			}
		}
		continue;
	fr_break_3:
		if (condition_o) {
			if (condition_r) {
				action_6;
			}
			else {
				action_2;
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_r) {
						action_6;
					}
					else {
						action_2;
					}
				}
				else {
					if (condition_r) {
						action_6;
					}
					else {
						action_2;
					}
				}
			}
			else {
				if (condition_p) {
					action_2;
				}
				else {
					if (condition_t) {
						action_2;
					}
					else {
						action_1;
					}
				}
			}
		}
		continue;
	fr_break_9:
		if (condition_o) {
			if (condition_n) {
				action_6;
			}
			else {
				if (condition_r) {
					action_6;
				}
				else {
					action_2;
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_n) {
						action_6;
					}
					else {
						if (condition_r) {
							action_6;
						}
						else {
							action_2;
						}
					}
				}
				else {
					if (condition_r) {
						action_6;
					}
					else {
						if (condition_n) {
							action_6;
						}
						else {
							action_2;
						}
					}
				}
			}
			else {
				if (condition_p) {
					action_2;
				}
				else {
					if (condition_t) {
						action_2;
					}
					else {
						action_1;
					}
				}
			}
		}
		continue;
	}

	for (int r = 2; r < h - 1; r += 2) {
		// Get row pointers
		const uchar* const img_row = img.ptr<uchar>(r);
		const uchar* const img_row_fol = (uchar *)(((char *)img_row) + img.step.p[0]);
		uint* const imgLabels_row = imgLabels.ptr<uint>(r);
		uint* const imgLabels_row_prev_prev = (uint *)(((char *)imgLabels_row) - imgLabels.step.p[0] - imgLabels.step.p[0]);

		int c = -2;
		finish_condition(0);
		if (condition_o) {
			if (condition_j) {
				if (condition_i) {
					action_4;
					goto tree_73;
				}
				else {
					action_4;
					goto tree_73;
				}
			}
			else {
				if (condition_p) {
					if (condition_k) {
						if (condition_i) {
							if (condition_d) {
								action_5;
								goto tree_4;
							}
							else {
								action_10;
								goto tree_4;
							}
						}
						else {
							action_5;
							goto tree_4;
						}
					}
					else {
						if (condition_i) {
							action_4;
							goto tree_3;
						}
						else {
							action_2;
							goto tree_2;
						}
					}
				}
				else {
					if (condition_i) {
						action_4;
						goto tree_63;
					}
					else {
						action_2;
						goto tree_61;
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					action_2;
					goto tree_8;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_0: finish_condition(0);
		if (condition_o) {
			if (condition_j) {
				if (condition_i) {
					action_4;
					goto tree_73;
				}
				else {
					if (condition_h) {
						if (condition_c) {
							action_4;
							goto tree_73;
						}
						else {
							action_7;
							goto tree_73;
						}
					}
					else {
						action_4;
						goto tree_73;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_k) {
						if (condition_i) {
							if (condition_d) {
								action_5;
								goto tree_4;
							}
							else {
								action_10;
								goto tree_4;
							}
						}
						else {
							if (condition_h) {
								if (condition_d) {
									if (condition_c) {
										action_5;
										goto tree_4;
									}
									else {
										action_8;
										goto tree_4;
									}
								}
								else {
									action_8;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
					}
					else {
						if (condition_i) {
							action_4;
							goto tree_3;
						}
						else {
							if (condition_h) {
								action_3;
								goto tree_2;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					if (condition_i) {
						action_4;
						goto tree_63;
					}
					else {
						if (condition_h) {
							action_3;
							goto tree_61;
						}
						else {
							action_2;
							goto tree_61;
						}
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					action_2;
					goto tree_8;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_1: finish_condition(1);
		if (condition_o) {
			if (condition_j) {
				if (condition_i) {
					action_11;
					goto tree_73;
				}
				else {
					if (condition_h) {
						if (condition_c) {
							action_11;
							goto tree_73;
						}
						else {
							action_14;
							goto tree_73;
						}
					}
					else {
						action_11;
						goto tree_73;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_k) {
						if (condition_i) {
							if (condition_d) {
								action_12;
								goto tree_4;
							}
							else {
								action_16;
								goto tree_4;
							}
						}
						else {
							if (condition_h) {
								if (condition_d) {
									if (condition_c) {
										action_12;
										goto tree_4;
									}
									else {
										action_15;
										goto tree_4;
									}
								}
								else {
									action_15;
									goto tree_4;
								}
							}
							else {
								action_12;
								goto tree_4;
							}
						}
					}
					else {
						if (condition_h) {
							action_9;
							goto tree_47;
						}
						else {
							if (condition_i) {
								action_11;
								goto tree_3;
							}
							else {
								action_6;
								goto tree_2;
							}
						}
					}
				}
				else {
					if (condition_h) {
						action_9;
						goto tree_82;
					}
					else {
						if (condition_i) {
							action_11;
							goto tree_63;
						}
						else {
							action_6;
							goto tree_61;
						}
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						action_11;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_d) {
								action_12;
								goto tree_4;
							}
							else {
								if (condition_i) {
									action_16;
									goto tree_4;
								}
								else {
									action_12;
									goto tree_4;
								}
							}
						}
						else {
							if (condition_i) {
								action_11;
								goto tree_3;
							}
							else {
								action_6;
								goto tree_2;
							}
						}
					}
				}
				else {
					action_6;
					goto tree_8;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_2: finish_condition(2);
		if (condition_o) {
			if (condition_j) {
				action_11;
				goto tree_73;
			}
			else {
				if (condition_p) {
					if (condition_k) {
						if (condition_d) {
							action_12;
							goto tree_4;
						}
						else {
							action_12;
							goto tree_4;
						}
					}
					else {
						action_6;
						goto tree_47;
					}
				}
				else {
					action_6;
					goto tree_82;
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						action_11;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_d) {
								action_12;
								goto tree_4;
							}
							else {
								action_12;
								goto tree_4;
							}
						}
						else {
							action_6;
							goto tree_47;
						}
					}
				}
				else {
					action_6;
					goto tree_8;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							action_5;
							goto tree_4;
						}
						else {
							action_2;
							goto tree_2;
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_3: finish_condition(3);
		if (condition_o) {
			if (condition_j) {
				if (condition_c) {
					if (condition_b) {
						action_6;
						goto tree_73;
					}
					else {
						action_11;
						goto tree_73;
					}
				}
				else {
					action_11;
					goto tree_73;
				}
			}
			else {
				if (condition_p) {
					if (condition_k) {
						if (condition_d) {
							if (condition_c) {
								if (condition_b) {
									action_6;
									goto tree_4;
								}
								else {
									action_12;
									goto tree_4;
								}
							}
							else {
								action_12;
								goto tree_4;
							}
						}
						else {
							action_12;
							goto tree_4;
						}
					}
					else {
						action_6;
						goto tree_47;
					}
				}
				else {
					action_6;
					goto tree_82;
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						if (condition_c) {
							if (condition_b) {
								action_6;
								goto tree_7;
							}
							else {
								action_11;
								goto tree_7;
							}
						}
						else {
							action_11;
							goto tree_7;
						}
					}
					else {
						if (condition_k) {
							if (condition_d) {
								if (condition_c) {
									if (condition_b) {
										action_6;
										goto tree_4;
									}
									else {
										action_12;
										goto tree_4;
									}
								}
								else {
									action_12;
									goto tree_4;
								}
							}
							else {
								action_12;
								goto tree_4;
							}
						}
						else {
							action_6;
							goto tree_47;
						}
					}
				}
				else {
					action_6;
					goto tree_8;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							action_5;
							goto tree_4;
						}
						else {
							action_2;
							goto tree_2;
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_4: finish_condition(4);
		if (condition_o) {
			if (condition_j) {
				action_6;
				goto tree_73;
			}
			else {
				if (condition_p) {
					if (condition_k) {
						if (condition_d) {
							action_6;
							goto tree_4;
						}
						else {
							action_12;
							goto tree_4;
						}
					}
					else {
						action_6;
						goto tree_47;
					}
				}
				else {
					action_6;
					goto tree_82;
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						action_6;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_d) {
								action_6;
								goto tree_4;
							}
							else {
								action_12;
								goto tree_4;
							}
						}
						else {
							action_6;
							goto tree_47;
						}
					}
				}
				else {
					action_6;
					goto tree_8;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_d) {
								action_5;
								goto tree_4;
							}
							else {
								action_10;
								goto tree_4;
							}
						}
						else {
							action_4;
							goto tree_3;
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_7: finish_condition(7);
		if (condition_o) {
			if (condition_j) {
				if (condition_i) {
					action_6;
					goto tree_73;
				}
				else {
					if (condition_c) {
						action_6;
						goto tree_73;
					}
					else {
						action_11;
						goto tree_73;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_k) {
						if (condition_d) {
							if (condition_i) {
								action_6;
								goto tree_4;
							}
							else {
								if (condition_c) {
									action_6;
									goto tree_4;
								}
								else {
									action_12;
									goto tree_4;
								}
							}
						}
						else {
							action_12;
							goto tree_4;
						}
					}
					else {
						action_6;
						goto tree_47;
					}
				}
				else {
					action_6;
					goto tree_82;
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						if (condition_i) {
							action_6;
							goto tree_7;
						}
						else {
							if (condition_c) {
								action_6;
								goto tree_7;
							}
							else {
								action_11;
								goto tree_7;
							}
						}
					}
					else {
						if (condition_k) {
							if (condition_d) {
								if (condition_i) {
									action_6;
									goto tree_4;
								}
								else {
									if (condition_c) {
										action_6;
										goto tree_4;
									}
									else {
										action_12;
										goto tree_4;
									}
								}
							}
							else {
								action_12;
								goto tree_4;
							}
						}
						else {
							action_6;
							goto tree_47;
						}
					}
				}
				else {
					action_6;
					goto tree_8;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_8: finish_condition(8);
		if (condition_o) {
			if (condition_r) {
				if (condition_j) {
					if (condition_i) {
						action_11;
						goto tree_73;
					}
					else {
						if (condition_h) {
							if (condition_c) {
								action_11;
								goto tree_73;
							}
							else {
								action_14;
								goto tree_73;
							}
						}
						else {
							action_11;
							goto tree_73;
						}
					}
				}
				else {
					if (condition_p) {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_12;
									goto tree_4;
								}
								else {
									action_16;
									goto tree_4;
								}
							}
							else {
								if (condition_h) {
									if (condition_d) {
										if (condition_c) {
											action_12;
											goto tree_4;
										}
										else {
											action_15;
											goto tree_4;
										}
									}
									else {
										action_15;
										goto tree_4;
									}
								}
								else {
									action_12;
									goto tree_4;
								}
							}
						}
						else {
							if (condition_h) {
								action_9;
								goto tree_47;
							}
							else {
								if (condition_i) {
									action_11;
									goto tree_3;
								}
								else {
									action_6;
									goto tree_2;
								}
							}
						}
					}
					else {
						if (condition_h) {
							action_9;
							goto tree_82;
						}
						else {
							if (condition_i) {
								action_11;
								goto tree_63;
							}
							else {
								action_6;
								goto tree_61;
							}
						}
					}
				}
			}
			else {
				if (condition_j) {
					if (condition_i) {
						action_4;
						goto tree_73;
					}
					else {
						if (condition_h) {
							if (condition_c) {
								action_4;
								goto tree_73;
							}
							else {
								action_7;
								goto tree_73;
							}
						}
						else {
							action_4;
							goto tree_73;
						}
					}
				}
				else {
					if (condition_p) {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								if (condition_h) {
									if (condition_d) {
										if (condition_c) {
											action_5;
											goto tree_4;
										}
										else {
											action_8;
											goto tree_4;
										}
									}
									else {
										action_8;
										goto tree_4;
									}
								}
								else {
									action_5;
									goto tree_4;
								}
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								if (condition_h) {
									action_3;
									goto tree_2;
								}
								else {
									action_2;
									goto tree_2;
								}
							}
						}
					}
					else {
						if (condition_i) {
							action_4;
							goto tree_63;
						}
						else {
							if (condition_h) {
								action_3;
								goto tree_61;
							}
							else {
								action_2;
								goto tree_61;
							}
						}
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_r) {
						if (condition_j) {
							action_11;
							goto tree_7;
						}
						else {
							if (condition_k) {
								if (condition_d) {
									action_12;
									goto tree_4;
								}
								else {
									if (condition_i) {
										action_16;
										goto tree_4;
									}
									else {
										action_12;
										goto tree_4;
									}
								}
							}
							else {
								if (condition_i) {
									action_11;
									goto tree_3;
								}
								else {
									action_6;
									goto tree_2;
								}
							}
						}
					}
					else {
						if (condition_j) {
							action_4;
							goto tree_7;
						}
						else {
							if (condition_k) {
								if (condition_i) {
									if (condition_d) {
										action_5;
										goto tree_4;
									}
									else {
										action_10;
										goto tree_4;
									}
								}
								else {
									action_5;
									goto tree_4;
								}
							}
							else {
								if (condition_i) {
									action_4;
									goto tree_3;
								}
								else {
									action_2;
									goto tree_2;
								}
							}
						}
					}
				}
				else {
					if (condition_r) {
						action_6;
						goto tree_8;
					}
					else {
						action_2;
						goto tree_8;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_47: finish_condition(47);
		if (condition_o) {
			if (condition_j) {
				if (condition_c) {
					if (condition_g) {
						if (condition_b) {
							action_6;
							goto tree_73;
						}
						else {
							action_11;
							goto tree_73;
						}
					}
					else {
						action_11;
						goto tree_73;
					}
				}
				else {
					action_11;
					goto tree_73;
				}
			}
			else {
				if (condition_p) {
					if (condition_k) {
						if (condition_d) {
							if (condition_c) {
								if (condition_g) {
									if (condition_b) {
										action_6;
										goto tree_4;
									}
									else {
										action_12;
										goto tree_4;
									}
								}
								else {
									action_12;
									goto tree_4;
								}
							}
							else {
								action_12;
								goto tree_4;
							}
						}
						else {
							action_12;
							goto tree_4;
						}
					}
					else {
						action_6;
						goto tree_47;
					}
				}
				else {
					action_6;
					goto tree_82;
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						if (condition_c) {
							if (condition_g) {
								if (condition_b) {
									action_6;
									goto tree_7;
								}
								else {
									action_11;
									goto tree_7;
								}
							}
							else {
								action_11;
								goto tree_7;
							}
						}
						else {
							action_11;
							goto tree_7;
						}
					}
					else {
						if (condition_k) {
							if (condition_d) {
								if (condition_c) {
									if (condition_g) {
										if (condition_b) {
											action_6;
											goto tree_4;
										}
										else {
											action_12;
											goto tree_4;
										}
									}
									else {
										action_12;
										goto tree_4;
									}
								}
								else {
									action_12;
									goto tree_4;
								}
							}
							else {
								action_12;
								goto tree_4;
							}
						}
						else {
							action_6;
							goto tree_47;
						}
					}
				}
				else {
					action_6;
					goto tree_8;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							action_5;
							goto tree_4;
						}
						else {
							action_2;
							goto tree_2;
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_61: finish_condition(61);
		if (condition_o) {
			if (condition_r) {
				if (condition_j) {
					action_11;
					goto tree_73;
				}
				else {
					if (condition_p) {
						if (condition_k) {
							if (condition_d) {
								action_12;
								goto tree_4;
							}
							else {
								if (condition_i) {
									action_16;
									goto tree_4;
								}
								else {
									action_12;
									goto tree_4;
								}
							}
						}
						else {
							if (condition_i) {
								action_11;
								goto tree_3;
							}
							else {
								action_6;
								goto tree_2;
							}
						}
					}
					else {
						if (condition_i) {
							action_11;
							goto tree_63;
						}
						else {
							action_6;
							goto tree_61;
						}
					}
				}
			}
			else {
				if (condition_j) {
					action_4;
					goto tree_73;
				}
				else {
					if (condition_p) {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
					else {
						if (condition_i) {
							action_4;
							goto tree_63;
						}
						else {
							action_2;
							goto tree_61;
						}
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_r) {
						if (condition_j) {
							action_11;
							goto tree_7;
						}
						else {
							if (condition_k) {
								if (condition_d) {
									action_12;
									goto tree_4;
								}
								else {
									if (condition_i) {
										action_16;
										goto tree_4;
									}
									else {
										action_12;
										goto tree_4;
									}
								}
							}
							else {
								if (condition_i) {
									action_11;
									goto tree_3;
								}
								else {
									action_6;
									goto tree_2;
								}
							}
						}
					}
					else {
						if (condition_j) {
							action_4;
							goto tree_7;
						}
						else {
							if (condition_k) {
								if (condition_i) {
									if (condition_d) {
										action_5;
										goto tree_4;
									}
									else {
										action_10;
										goto tree_4;
									}
								}
								else {
									action_5;
									goto tree_4;
								}
							}
							else {
								if (condition_i) {
									action_4;
									goto tree_3;
								}
								else {
									action_2;
									goto tree_2;
								}
							}
						}
					}
				}
				else {
					if (condition_r) {
						action_6;
						goto tree_8;
					}
					else {
						action_2;
						goto tree_8;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_63: finish_condition(63);
		if (condition_o) {
			if (condition_r) {
				if (condition_j) {
					if (condition_b) {
						if (condition_i) {
							action_6;
							goto tree_73;
						}
						else {
							if (condition_c) {
								action_6;
								goto tree_73;
							}
							else {
								action_11;
								goto tree_73;
							}
						}
					}
					else {
						action_11;
						goto tree_73;
					}
				}
				else {
					if (condition_p) {
						if (condition_k) {
							if (condition_d) {
								if (condition_b) {
									if (condition_i) {
										action_6;
										goto tree_4;
									}
									else {
										if (condition_c) {
											action_6;
											goto tree_4;
										}
										else {
											action_12;
											goto tree_4;
										}
									}
								}
								else {
									action_12;
									goto tree_4;
								}
							}
							else {
								if (condition_i) {
									if (condition_b) {
										action_12;
										goto tree_4;
									}
									else {
										action_16;
										goto tree_4;
									}
								}
								else {
									action_12;
									goto tree_4;
								}
							}
						}
						else {
							if (condition_i) {
								if (condition_b) {
									action_6;
									goto tree_3;
								}
								else {
									action_11;
									goto tree_3;
								}
							}
							else {
								action_6;
								goto tree_2;
							}
						}
					}
					else {
						if (condition_i) {
							if (condition_b) {
								action_6;
								goto tree_63;
							}
							else {
								action_11;
								goto tree_63;
							}
						}
						else {
							action_6;
							goto tree_61;
						}
					}
				}
			}
			else {
				if (condition_j) {
					action_4;
					goto tree_73;
				}
				else {
					if (condition_p) {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
					else {
						if (condition_i) {
							action_4;
							goto tree_63;
						}
						else {
							action_2;
							goto tree_61;
						}
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_r) {
						if (condition_j) {
							if (condition_b) {
								if (condition_i) {
									action_6;
									goto tree_7;
								}
								else {
									if (condition_c) {
										action_6;
										goto tree_7;
									}
									else {
										action_11;
										goto tree_7;
									}
								}
							}
							else {
								action_11;
								goto tree_7;
							}
						}
						else {
							if (condition_k) {
								if (condition_d) {
									if (condition_b) {
										if (condition_i) {
											action_6;
											goto tree_4;
										}
										else {
											if (condition_c) {
												action_6;
												goto tree_4;
											}
											else {
												action_12;
												goto tree_4;
											}
										}
									}
									else {
										action_12;
										goto tree_4;
									}
								}
								else {
									if (condition_i) {
										if (condition_b) {
											action_12;
											goto tree_4;
										}
										else {
											action_16;
											goto tree_4;
										}
									}
									else {
										action_12;
										goto tree_4;
									}
								}
							}
							else {
								if (condition_i) {
									if (condition_b) {
										action_6;
										goto tree_3;
									}
									else {
										action_11;
										goto tree_3;
									}
								}
								else {
									action_6;
									goto tree_2;
								}
							}
						}
					}
					else {
						if (condition_j) {
							action_4;
							goto tree_7;
						}
						else {
							if (condition_k) {
								if (condition_i) {
									if (condition_d) {
										action_5;
										goto tree_4;
									}
									else {
										action_10;
										goto tree_4;
									}
								}
								else {
									action_5;
									goto tree_4;
								}
							}
							else {
								if (condition_i) {
									action_4;
									goto tree_3;
								}
								else {
									action_2;
									goto tree_2;
								}
							}
						}
					}
				}
				else {
					if (condition_r) {
						action_6;
						goto tree_8;
					}
					else {
						action_2;
						goto tree_8;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_73: finish_condition(73);
		if (condition_o) {
			if (condition_n) {
				if (condition_j) {
					if (condition_i) {
						action_6;
						goto tree_73;
					}
					else {
						if (condition_c) {
							action_6;
							goto tree_73;
						}
						else {
							action_11;
							goto tree_73;
						}
					}
				}
				else {
					if (condition_p) {
						if (condition_k) {
							if (condition_d) {
								if (condition_i) {
									action_6;
									goto tree_4;
								}
								else {
									if (condition_c) {
										action_6;
										goto tree_4;
									}
									else {
										action_12;
										goto tree_4;
									}
								}
							}
							else {
								action_12;
								goto tree_4;
							}
						}
						else {
							action_6;
							goto tree_47;
						}
					}
					else {
						action_6;
						goto tree_82;
					}
				}
			}
			else {
				if (condition_r) {
					if (condition_j) {
						if (condition_i) {
							action_6;
							goto tree_73;
						}
						else {
							if (condition_c) {
								action_6;
								goto tree_73;
							}
							else {
								action_11;
								goto tree_73;
							}
						}
					}
					else {
						if (condition_p) {
							if (condition_k) {
								if (condition_d) {
									if (condition_i) {
										action_6;
										goto tree_4;
									}
									else {
										if (condition_c) {
											action_6;
											goto tree_4;
										}
										else {
											action_12;
											goto tree_4;
										}
									}
								}
								else {
									action_12;
									goto tree_4;
								}
							}
							else {
								action_6;
								goto tree_47;
							}
						}
						else {
							action_6;
							goto tree_82;
						}
					}
				}
				else {
					if (condition_j) {
						if (condition_i) {
							action_4;
							goto tree_73;
						}
						else {
							if (condition_c) {
								action_4;
								goto tree_73;
							}
							else {
								action_7;
								goto tree_73;
							}
						}
					}
					else {
						if (condition_p) {
							if (condition_k) {
								if (condition_i) {
									if (condition_d) {
										action_5;
										goto tree_4;
									}
									else {
										action_10;
										goto tree_4;
									}
								}
								else {
									if (condition_d) {
										if (condition_c) {
											action_5;
											goto tree_4;
										}
										else {
											action_8;
											goto tree_4;
										}
									}
									else {
										action_8;
										goto tree_4;
									}
								}
							}
							else {
								if (condition_i) {
									action_4;
									goto tree_3;
								}
								else {
									action_3;
									goto tree_2;
								}
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_63;
							}
							else {
								action_3;
								goto tree_61;
							}
						}
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_n) {
						if (condition_j) {
							if (condition_i) {
								action_6;
								goto tree_7;
							}
							else {
								if (condition_c) {
									action_6;
									goto tree_7;
								}
								else {
									action_11;
									goto tree_7;
								}
							}
						}
						else {
							if (condition_k) {
								if (condition_d) {
									if (condition_i) {
										action_6;
										goto tree_4;
									}
									else {
										if (condition_c) {
											action_6;
											goto tree_4;
										}
										else {
											action_12;
											goto tree_4;
										}
									}
								}
								else {
									action_12;
									goto tree_4;
								}
							}
							else {
								action_6;
								goto tree_47;
							}
						}
					}
					else {
						if (condition_r) {
							if (condition_j) {
								if (condition_i) {
									action_6;
									goto tree_7;
								}
								else {
									if (condition_c) {
										action_6;
										goto tree_7;
									}
									else {
										action_11;
										goto tree_7;
									}
								}
							}
							else {
								if (condition_k) {
									if (condition_d) {
										if (condition_i) {
											action_6;
											goto tree_4;
										}
										else {
											if (condition_c) {
												action_6;
												goto tree_4;
											}
											else {
												action_12;
												goto tree_4;
											}
										}
									}
									else {
										action_12;
										goto tree_4;
									}
								}
								else {
									if (condition_i) {
										action_6;
										goto tree_3;
									}
									else {
										action_6;
										goto tree_2;
									}
								}
							}
						}
						else {
							if (condition_j) {
								action_4;
								goto tree_7;
							}
							else {
								if (condition_k) {
									if (condition_i) {
										if (condition_d) {
											action_5;
											goto tree_4;
										}
										else {
											action_10;
											goto tree_4;
										}
									}
									else {
										action_5;
										goto tree_4;
									}
								}
								else {
									if (condition_i) {
										action_4;
										goto tree_3;
									}
									else {
										action_2;
										goto tree_2;
									}
								}
							}
						}
					}
				}
				else {
					if (condition_r) {
						action_6;
						goto tree_8;
					}
					else {
						if (condition_n) {
							action_6;
							goto tree_8;
						}
						else {
							action_2;
							goto tree_8;
						}
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_82: finish_condition(82);
		if (condition_o) {
			if (condition_r) {
				if (condition_j) {
					if (condition_g) {
						if (condition_b) {
							if (condition_i) {
								action_6;
								goto tree_73;
							}
							else {
								if (condition_c) {
									action_6;
									goto tree_73;
								}
								else {
									action_11;
									goto tree_73;
								}
							}
						}
						else {
							action_11;
							goto tree_73;
						}
					}
					else {
						action_11;
						goto tree_73;
					}
				}
				else {
					if (condition_p) {
						if (condition_k) {
							if (condition_d) {
								if (condition_g) {
									if (condition_b) {
										if (condition_i) {
											action_6;
											goto tree_4;
										}
										else {
											if (condition_c) {
												action_6;
												goto tree_4;
											}
											else {
												action_12;
												goto tree_4;
											}
										}
									}
									else {
										action_12;
										goto tree_4;
									}
								}
								else {
									action_12;
									goto tree_4;
								}
							}
							else {
								if (condition_i) {
									if (condition_g) {
										if (condition_b) {
											action_12;
											goto tree_4;
										}
										else {
											action_16;
											goto tree_4;
										}
									}
									else {
										action_16;
										goto tree_4;
									}
								}
								else {
									action_12;
									goto tree_4;
								}
							}
						}
						else {
							if (condition_i) {
								if (condition_g) {
									if (condition_b) {
										action_6;
										goto tree_3;
									}
									else {
										action_11;
										goto tree_3;
									}
								}
								else {
									action_11;
									goto tree_3;
								}
							}
							else {
								action_6;
								goto tree_2;
							}
						}
					}
					else {
						if (condition_i) {
							if (condition_g) {
								if (condition_b) {
									action_6;
									goto tree_63;
								}
								else {
									action_11;
									goto tree_63;
								}
							}
							else {
								action_11;
								goto tree_63;
							}
						}
						else {
							action_6;
							goto tree_61;
						}
					}
				}
			}
			else {
				if (condition_j) {
					action_4;
					goto tree_73;
				}
				else {
					if (condition_p) {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
					else {
						if (condition_i) {
							action_4;
							goto tree_63;
						}
						else {
							action_2;
							goto tree_61;
						}
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_r) {
						if (condition_j) {
							if (condition_g) {
								if (condition_b) {
									if (condition_i) {
										action_6;
										goto tree_7;
									}
									else {
										if (condition_c) {
											action_6;
											goto tree_7;
										}
										else {
											action_11;
											goto tree_7;
										}
									}
								}
								else {
									action_11;
									goto tree_7;
								}
							}
							else {
								action_11;
								goto tree_7;
							}
						}
						else {
							if (condition_k) {
								if (condition_d) {
									if (condition_g) {
										if (condition_b) {
											if (condition_i) {
												action_6;
												goto tree_4;
											}
											else {
												if (condition_c) {
													action_6;
													goto tree_4;
												}
												else {
													action_12;
													goto tree_4;
												}
											}
										}
										else {
											action_12;
											goto tree_4;
										}
									}
									else {
										action_12;
										goto tree_4;
									}
								}
								else {
									if (condition_i) {
										if (condition_g) {
											if (condition_b) {
												action_12;
												goto tree_4;
											}
											else {
												action_16;
												goto tree_4;
											}
										}
										else {
											action_16;
											goto tree_4;
										}
									}
									else {
										action_12;
										goto tree_4;
									}
								}
							}
							else {
								if (condition_i) {
									if (condition_g) {
										if (condition_b) {
											action_6;
											goto tree_3;
										}
										else {
											action_11;
											goto tree_3;
										}
									}
									else {
										action_11;
										goto tree_3;
									}
								}
								else {
									action_6;
									goto tree_2;
								}
							}
						}
					}
					else {
						if (condition_j) {
							action_4;
							goto tree_7;
						}
						else {
							if (condition_k) {
								if (condition_i) {
									if (condition_d) {
										action_5;
										goto tree_4;
									}
									else {
										action_10;
										goto tree_4;
									}
								}
								else {
									action_5;
									goto tree_4;
								}
							}
							else {
								if (condition_i) {
									action_4;
									goto tree_3;
								}
								else {
									action_2;
									goto tree_2;
								}
							}
						}
					}
				}
				else {
					if (condition_r) {
						action_6;
						goto tree_8;
					}
					else {
						action_2;
						goto tree_8;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	break_0:
		if (condition_o) {
			if (condition_j) {
				if (condition_i) {
					action_4;
				}
				else {
					if (condition_h) {
						if (condition_c) {
							action_4;
						}
						else {
							action_7;
						}
					}
					else {
						action_4;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_i) {
						action_4;
					}
					else {
						if (condition_h) {
							action_3;
						}
						else {
							action_2;
						}
					}
				}
				else {
					if (condition_i) {
						action_4;
					}
					else {
						if (condition_h) {
							action_3;
						}
						else {
							action_2;
						}
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						action_4;
					}
					else {
						if (condition_i) {
							action_4;
						}
						else {
							action_2;
						}
					}
				}
				else {
					action_2;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
					}
					else {
						if (condition_i) {
							action_4;
						}
						else {
							action_2;
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
					}
					else {
						action_1;
					}
				}
			}
		}
		continue;
	break_1:
		if (condition_o) {
			if (condition_j) {
				if (condition_i) {
					action_11;
				}
				else {
					if (condition_h) {
						if (condition_c) {
							action_11;
						}
						else {
							action_14;
						}
					}
					else {
						action_11;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_h) {
						action_9;
					}
					else {
						if (condition_i) {
							action_11;
						}
						else {
							action_6;
						}
					}
				}
				else {
					if (condition_h) {
						action_9;
					}
					else {
						if (condition_i) {
							action_11;
						}
						else {
							action_6;
						}
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						action_11;
					}
					else {
						if (condition_i) {
							action_11;
						}
						else {
							action_6;
						}
					}
				}
				else {
					action_6;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
					}
					else {
						if (condition_i) {
							action_4;
						}
						else {
							action_2;
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
					}
					else {
						action_1;
					}
				}
			}
		}
		continue;
	break_2:
		if (condition_o) {
			if (condition_j) {
				action_11;
			}
			else {
				if (condition_p) {
					action_6;
				}
				else {
					action_6;
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						action_11;
					}
					else {
						action_6;
					}
				}
				else {
					action_6;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
					}
					else {
						action_2;
					}
				}
				else {
					if (condition_t) {
						action_2;
					}
					else {
						action_1;
					}
				}
			}
		}
		continue;
	break_3:
		if (condition_o) {
			if (condition_j) {
				if (condition_c) {
					if (condition_b) {
						action_6;
					}
					else {
						action_11;
					}
				}
				else {
					action_11;
				}
			}
			else {
				if (condition_p) {
					action_6;
				}
				else {
					action_6;
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						if (condition_c) {
							if (condition_b) {
								action_6;
							}
							else {
								action_11;
							}
						}
						else {
							action_11;
						}
					}
					else {
						action_6;
					}
				}
				else {
					action_6;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
					}
					else {
						action_2;
					}
				}
				else {
					if (condition_t) {
						action_2;
					}
					else {
						action_1;
					}
				}
			}
		}
		continue;
	break_4:
		if (condition_o) {
			if (condition_j) {
				action_6;
			}
			else {
				if (condition_p) {
					action_6;
				}
				else {
					action_6;
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						action_6;
					}
					else {
						action_6;
					}
				}
				else {
					action_6;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
					}
					else {
						action_4;
					}
				}
				else {
					if (condition_t) {
						action_2;
					}
					else {
						action_1;
					}
				}
			}
		}
		continue;
	break_7:
		if (condition_o) {
			if (condition_j) {
				if (condition_i) {
					action_6;
				}
				else {
					if (condition_c) {
						action_6;
					}
					else {
						action_11;
					}
				}
			}
			else {
				if (condition_p) {
					action_6;
				}
				else {
					action_6;
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						if (condition_i) {
							action_6;
						}
						else {
							if (condition_c) {
								action_6;
							}
							else {
								action_11;
							}
						}
					}
					else {
						action_6;
					}
				}
				else {
					action_6;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
					}
					else {
						if (condition_i) {
							action_4;
						}
						else {
							action_2;
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
					}
					else {
						action_1;
					}
				}
			}
		}
		continue;
	break_8:
		if (condition_o) {
			if (condition_r) {
				if (condition_j) {
					if (condition_i) {
						action_11;
					}
					else {
						if (condition_h) {
							if (condition_c) {
								action_11;
							}
							else {
								action_14;
							}
						}
						else {
							action_11;
						}
					}
				}
				else {
					if (condition_p) {
						if (condition_h) {
							action_9;
						}
						else {
							if (condition_i) {
								action_11;
							}
							else {
								action_6;
							}
						}
					}
					else {
						if (condition_h) {
							action_9;
						}
						else {
							if (condition_i) {
								action_11;
							}
							else {
								action_6;
							}
						}
					}
				}
			}
			else {
				if (condition_j) {
					if (condition_i) {
						action_4;
					}
					else {
						if (condition_h) {
							if (condition_c) {
								action_4;
							}
							else {
								action_7;
							}
						}
						else {
							action_4;
						}
					}
				}
				else {
					if (condition_p) {
						if (condition_i) {
							action_4;
						}
						else {
							if (condition_h) {
								action_3;
							}
							else {
								action_2;
							}
						}
					}
					else {
						if (condition_i) {
							action_4;
						}
						else {
							if (condition_h) {
								action_3;
							}
							else {
								action_2;
							}
						}
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_r) {
						if (condition_j) {
							action_11;
						}
						else {
							if (condition_i) {
								action_11;
							}
							else {
								action_6;
							}
						}
					}
					else {
						if (condition_j) {
							action_4;
						}
						else {
							if (condition_i) {
								action_4;
							}
							else {
								action_2;
							}
						}
					}
				}
				else {
					if (condition_r) {
						action_6;
					}
					else {
						action_2;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
					}
					else {
						if (condition_i) {
							action_4;
						}
						else {
							action_2;
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
					}
					else {
						action_1;
					}
				}
			}
		}
		continue;
	break_47:
		if (condition_o) {
			if (condition_j) {
				if (condition_c) {
					if (condition_g) {
						if (condition_b) {
							action_6;
						}
						else {
							action_11;
						}
					}
					else {
						action_11;
					}
				}
				else {
					action_11;
				}
			}
			else {
				if (condition_p) {
					action_6;
				}
				else {
					action_6;
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						if (condition_c) {
							if (condition_g) {
								if (condition_b) {
									action_6;
								}
								else {
									action_11;
								}
							}
							else {
								action_11;
							}
						}
						else {
							action_11;
						}
					}
					else {
						action_6;
					}
				}
				else {
					action_6;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
					}
					else {
						action_2;
					}
				}
				else {
					if (condition_t) {
						action_2;
					}
					else {
						action_1;
					}
				}
			}
		}
		continue;
	break_61:
		if (condition_o) {
			if (condition_r) {
				if (condition_j) {
					action_11;
				}
				else {
					if (condition_p) {
						if (condition_i) {
							action_11;
						}
						else {
							action_6;
						}
					}
					else {
						if (condition_i) {
							action_11;
						}
						else {
							action_6;
						}
					}
				}
			}
			else {
				if (condition_j) {
					action_4;
				}
				else {
					if (condition_p) {
						if (condition_i) {
							action_4;
						}
						else {
							action_2;
						}
					}
					else {
						if (condition_i) {
							action_4;
						}
						else {
							action_2;
						}
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_r) {
						if (condition_j) {
							action_11;
						}
						else {
							if (condition_i) {
								action_11;
							}
							else {
								action_6;
							}
						}
					}
					else {
						if (condition_j) {
							action_4;
						}
						else {
							if (condition_i) {
								action_4;
							}
							else {
								action_2;
							}
						}
					}
				}
				else {
					if (condition_r) {
						action_6;
					}
					else {
						action_2;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
					}
					else {
						if (condition_i) {
							action_4;
						}
						else {
							action_2;
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
					}
					else {
						action_1;
					}
				}
			}
		}
		continue;
	break_63:
		if (condition_o) {
			if (condition_r) {
				if (condition_j) {
					if (condition_b) {
						if (condition_i) {
							action_6;
						}
						else {
							if (condition_c) {
								action_6;
							}
							else {
								action_11;
							}
						}
					}
					else {
						action_11;
					}
				}
				else {
					if (condition_p) {
						if (condition_i) {
							if (condition_b) {
								action_6;
							}
							else {
								action_11;
							}
						}
						else {
							action_6;
						}
					}
					else {
						if (condition_i) {
							if (condition_b) {
								action_6;
							}
							else {
								action_11;
							}
						}
						else {
							action_6;
						}
					}
				}
			}
			else {
				if (condition_j) {
					action_4;
				}
				else {
					if (condition_p) {
						if (condition_i) {
							action_4;
						}
						else {
							action_2;
						}
					}
					else {
						if (condition_i) {
							action_4;
						}
						else {
							action_2;
						}
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_r) {
						if (condition_j) {
							if (condition_b) {
								if (condition_i) {
									action_6;
								}
								else {
									if (condition_c) {
										action_6;
									}
									else {
										action_11;
									}
								}
							}
							else {
								action_11;
							}
						}
						else {
							if (condition_i) {
								if (condition_b) {
									action_6;
								}
								else {
									action_11;
								}
							}
							else {
								action_6;
							}
						}
					}
					else {
						if (condition_j) {
							action_4;
						}
						else {
							if (condition_i) {
								action_4;
							}
							else {
								action_2;
							}
						}
					}
				}
				else {
					if (condition_r) {
						action_6;
					}
					else {
						action_2;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
					}
					else {
						if (condition_i) {
							action_4;
						}
						else {
							action_2;
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
					}
					else {
						action_1;
					}
				}
			}
		}
		continue;
	break_73:
		if (condition_o) {
			if (condition_n) {
				if (condition_j) {
					if (condition_i) {
						action_6;
					}
					else {
						if (condition_c) {
							action_6;
						}
						else {
							action_11;
						}
					}
				}
				else {
					if (condition_p) {
						action_6;
					}
					else {
						action_6;
					}
				}
			}
			else {
				if (condition_r) {
					if (condition_j) {
						if (condition_i) {
							action_6;
						}
						else {
							if (condition_c) {
								action_6;
							}
							else {
								action_11;
							}
						}
					}
					else {
						if (condition_p) {
							action_6;
						}
						else {
							action_6;
						}
					}
				}
				else {
					if (condition_j) {
						if (condition_i) {
							action_4;
						}
						else {
							if (condition_c) {
								action_4;
							}
							else {
								action_7;
							}
						}
					}
					else {
						if (condition_p) {
							if (condition_i) {
								action_4;
							}
							else {
								action_3;
							}
						}
						else {
							if (condition_i) {
								action_4;
							}
							else {
								action_3;
							}
						}
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_n) {
						if (condition_j) {
							if (condition_i) {
								action_6;
							}
							else {
								if (condition_c) {
									action_6;
								}
								else {
									action_11;
								}
							}
						}
						else {
							action_6;
						}
					}
					else {
						if (condition_r) {
							if (condition_j) {
								if (condition_i) {
									action_6;
								}
								else {
									if (condition_c) {
										action_6;
									}
									else {
										action_11;
									}
								}
							}
							else {
								if (condition_i) {
									action_6;
								}
								else {
									action_6;
								}
							}
						}
						else {
							if (condition_j) {
								action_4;
							}
							else {
								if (condition_i) {
									action_4;
								}
								else {
									action_2;
								}
							}
						}
					}
				}
				else {
					if (condition_r) {
						action_6;
					}
					else {
						if (condition_n) {
							action_6;
						}
						else {
							action_2;
						}
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
					}
					else {
						if (condition_i) {
							action_4;
						}
						else {
							action_2;
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
					}
					else {
						action_1;
					}
				}
			}
		}
		continue;
	break_82:
		if (condition_o) {
			if (condition_r) {
				if (condition_j) {
					if (condition_g) {
						if (condition_b) {
							if (condition_i) {
								action_6;
							}
							else {
								if (condition_c) {
									action_6;
								}
								else {
									action_11;
								}
							}
						}
						else {
							action_11;
						}
					}
					else {
						action_11;
					}
				}
				else {
					if (condition_p) {
						if (condition_i) {
							if (condition_g) {
								if (condition_b) {
									action_6;
								}
								else {
									action_11;
								}
							}
							else {
								action_11;
							}
						}
						else {
							action_6;
						}
					}
					else {
						if (condition_i) {
							if (condition_g) {
								if (condition_b) {
									action_6;
								}
								else {
									action_11;
								}
							}
							else {
								action_11;
							}
						}
						else {
							action_6;
						}
					}
				}
			}
			else {
				if (condition_j) {
					action_4;
				}
				else {
					if (condition_p) {
						if (condition_i) {
							action_4;
						}
						else {
							action_2;
						}
					}
					else {
						if (condition_i) {
							action_4;
						}
						else {
							action_2;
						}
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_r) {
						if (condition_j) {
							if (condition_g) {
								if (condition_b) {
									if (condition_i) {
										action_6;
									}
									else {
										if (condition_c) {
											action_6;
										}
										else {
											action_11;
										}
									}
								}
								else {
									action_11;
								}
							}
							else {
								action_11;
							}
						}
						else {
							if (condition_i) {
								if (condition_g) {
									if (condition_b) {
										action_6;
									}
									else {
										action_11;
									}
								}
								else {
									action_11;
								}
							}
							else {
								action_6;
							}
						}
					}
					else {
						if (condition_j) {
							action_4;
						}
						else {
							if (condition_i) {
								action_4;
							}
							else {
								action_2;
							}
						}
					}
				}
				else {
					if (condition_r) {
						action_6;
					}
					else {
						action_2;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
					}
					else {
						if (condition_i) {
							action_4;
						}
						else {
							action_2;
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
					}
					else {
						action_1;
					}
				}
			}
		}
		continue;
	}

#undef fr_finish_condition
#undef finish_condition
}

inline static
void firstScanSpaghetti_oe(const Mat1b &img, Mat1i& imgLabels, uint* P, uint &lunique) {
	int w(img.cols), h(img.rows);
	int pix = 0, old_pix;

#define fr_finish_condition(n)	\
	c += 2; \
	old_pix = pix; \
	pix  = (img_row[c] > 0) << 31; \
	pix |= (img_row_fol[c] > 0) << 29; \
	pix |= (img_row[c + 1] > 0) << 30; \
	pix |= (img_row_fol[c + 1] > 0) << 28; \
	if (c == w - 2) \
		goto fr_break_##n; 
#define finish_condition(n) \
	c += 2; \
	old_pix = pix; \
	pix = (img_row[c] > 0) << 31; \
	pix |= (img_row_fol[c] > 0) << 29; \
	pix |= (img_row[c + 1] > 0) << 30; \
	pix |= (img_row_fol[c + 1] > 0) << 28; \
	if (c == w - 2) \
		goto break_##n;
#define lr_finish_condition(n) \
	c += 2; \
	old_pix = pix; \
	pix = (img_row[c] > 0) << 31; \
	pix |= (img_row[c + 1] > 0) << 30; \
	if (c == w - 2) \
		goto lr_break_##n;

	for (int r = 0; r < 2; r += 2) {
		// Get row pointers
		const uchar* const img_row = img.ptr<uchar>(r);
		const uchar* const img_row_fol = (uchar *)(((char *)img_row) + img.step.p[0]);
		uint* const imgLabels_row = imgLabels.ptr<uint>(r);

		int c = -2;
		fr_finish_condition(0);
		if (condition_o) {
			action_2;
			goto fr_tree_9;
		}
		else {
			if (condition_s) {
				if (condition_p) {
					action_2;
					goto fr_tree_1;
				}
				else {
					action_2;
					goto fr_tree_3;
				}
			}
			else {
				if (condition_p) {
					action_2;
					goto fr_tree_1;
				}
				else {
					if (condition_t) {
						action_2;
						goto fr_tree_1;
					}
					else {
						action_1;
						goto fr_tree_0;
					}
				}
			}
		}
	fr_tree_0: fr_finish_condition(0);
		if (condition_o) {
			action_2;
			goto fr_tree_9;
		}
		else {
			if (condition_s) {
				if (condition_p) {
					action_2;
					goto fr_tree_1;
				}
				else {
					action_2;
					goto fr_tree_3;
				}
			}
			else {
				if (condition_p) {
					action_2;
					goto fr_tree_1;
				}
				else {
					if (condition_t) {
						action_2;
						goto fr_tree_1;
					}
					else {
						action_1;
						goto fr_tree_0;
					}
				}
			}
		}
	fr_tree_1: fr_finish_condition(1);
		if (condition_o) {
			action_6;
			goto fr_tree_9;
		}
		else {
			if (condition_s) {
				if (condition_p) {
					action_6;
					goto fr_tree_1;
				}
				else {
					action_6;
					goto fr_tree_3;
				}
			}
			else {
				if (condition_p) {
					action_2;
					goto fr_tree_1;
				}
				else {
					if (condition_t) {
						action_2;
						goto fr_tree_1;
					}
					else {
						action_1;
						goto fr_tree_0;
					}
				}
			}
		}
	fr_tree_3: fr_finish_condition(3);
		if (condition_o) {
			if (condition_r) {
				action_6;
				goto fr_tree_9;
			}
			else {
				action_2;
				goto fr_tree_9;
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_r) {
						action_6;
						goto fr_tree_1;
					}
					else {
						action_2;
						goto fr_tree_1;
					}
				}
				else {
					if (condition_r) {
						action_6;
						goto fr_tree_3;
					}
					else {
						action_2;
						goto fr_tree_3;
					}
				}
			}
			else {
				if (condition_p) {
					action_2;
					goto fr_tree_1;
				}
				else {
					if (condition_t) {
						action_2;
						goto fr_tree_1;
					}
					else {
						action_1;
						goto fr_tree_0;
					}
				}
			}
		}
	fr_tree_9: fr_finish_condition(9);
		if (condition_o) {
			if (condition_n) {
				action_6;
				goto fr_tree_9;
			}
			else {
				if (condition_r) {
					action_6;
					goto fr_tree_9;
				}
				else {
					action_2;
					goto fr_tree_9;
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_n) {
						action_6;
						goto fr_tree_1;
					}
					else {
						if (condition_r) {
							action_6;
							goto fr_tree_1;
						}
						else {
							action_2;
							goto fr_tree_1;
						}
					}
				}
				else {
					if (condition_r) {
						action_6;
						goto fr_tree_3;
					}
					else {
						if (condition_n) {
							action_6;
							goto fr_tree_3;
						}
						else {
							action_2;
							goto fr_tree_3;
						}
					}
				}
			}
			else {
				if (condition_p) {
					action_2;
					goto fr_tree_1;
				}
				else {
					if (condition_t) {
						action_2;
						goto fr_tree_1;
					}
					else {
						action_1;
						goto fr_tree_0;
					}
				}
			}
		}
	fr_break_0:
		if (condition_o) {
			action_2;
		}
		else {
			if (condition_s) {
				if (condition_p) {
					action_2;
				}
				else {
					action_2;
				}
			}
			else {
				if (condition_p) {
					action_2;
				}
				else {
					if (condition_t) {
						action_2;
					}
					else {
						action_1;
					}
				}
			}
		}
		continue;
	fr_break_1:
		if (condition_o) {
			action_6;
		}
		else {
			if (condition_s) {
				if (condition_p) {
					action_6;
				}
				else {
					action_6;
				}
			}
			else {
				if (condition_p) {
					action_2;
				}
				else {
					if (condition_t) {
						action_2;
					}
					else {
						action_1;
					}
				}
			}
		}
		continue;
	fr_break_3:
		if (condition_o) {
			if (condition_r) {
				action_6;
			}
			else {
				action_2;
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_r) {
						action_6;
					}
					else {
						action_2;
					}
				}
				else {
					if (condition_r) {
						action_6;
					}
					else {
						action_2;
					}
				}
			}
			else {
				if (condition_p) {
					action_2;
				}
				else {
					if (condition_t) {
						action_2;
					}
					else {
						action_1;
					}
				}
			}
		}
		continue;
	fr_break_9:
		if (condition_o) {
			if (condition_n) {
				action_6;
			}
			else {
				if (condition_r) {
					action_6;
				}
				else {
					action_2;
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_n) {
						action_6;
					}
					else {
						if (condition_r) {
							action_6;
						}
						else {
							action_2;
						}
					}
				}
				else {
					if (condition_r) {
						action_6;
					}
					else {
						if (condition_n) {
							action_6;
						}
						else {
							action_2;
						}
					}
				}
			}
			else {
				if (condition_p) {
					action_2;
				}
				else {
					if (condition_t) {
						action_2;
					}
					else {
						action_1;
					}
				}
			}
		}
		continue;
	}

	for (int r = 2; r < h - 1; r += 2) {
		// Get row pointers
		const uchar* const img_row = img.ptr<uchar>(r);
		const uchar* const img_row_fol = (uchar *)(((char *)img_row) + img.step.p[0]);
		uint* const imgLabels_row = imgLabels.ptr<uint>(r);
		uint* const imgLabels_row_prev_prev = (uint *)(((char *)imgLabels_row) - imgLabels.step.p[0] - imgLabels.step.p[0]);

		int c = -2;
		finish_condition(0);
		if (condition_o) {
			if (condition_j) {
				if (condition_i) {
					action_4;
					goto tree_73;
				}
				else {
					action_4;
					goto tree_73;
				}
			}
			else {
				if (condition_p) {
					if (condition_k) {
						if (condition_i) {
							if (condition_d) {
								action_5;
								goto tree_4;
							}
							else {
								action_10;
								goto tree_4;
							}
						}
						else {
							action_5;
							goto tree_4;
						}
					}
					else {
						if (condition_i) {
							action_4;
							goto tree_3;
						}
						else {
							action_2;
							goto tree_2;
						}
					}
				}
				else {
					if (condition_i) {
						action_4;
						goto tree_63;
					}
					else {
						action_2;
						goto tree_61;
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					action_2;
					goto tree_8;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_0: finish_condition(0);
		if (condition_o) {
			if (condition_j) {
				if (condition_i) {
					action_4;
					goto tree_73;
				}
				else {
					if (condition_h) {
						if (condition_c) {
							action_4;
							goto tree_73;
						}
						else {
							action_7;
							goto tree_73;
						}
					}
					else {
						action_4;
						goto tree_73;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_k) {
						if (condition_i) {
							if (condition_d) {
								action_5;
								goto tree_4;
							}
							else {
								action_10;
								goto tree_4;
							}
						}
						else {
							if (condition_h) {
								if (condition_d) {
									if (condition_c) {
										action_5;
										goto tree_4;
									}
									else {
										action_8;
										goto tree_4;
									}
								}
								else {
									action_8;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
					}
					else {
						if (condition_i) {
							action_4;
							goto tree_3;
						}
						else {
							if (condition_h) {
								action_3;
								goto tree_2;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					if (condition_i) {
						action_4;
						goto tree_63;
					}
					else {
						if (condition_h) {
							action_3;
							goto tree_61;
						}
						else {
							action_2;
							goto tree_61;
						}
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					action_2;
					goto tree_8;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_1: finish_condition(1);
		if (condition_o) {
			if (condition_j) {
				if (condition_i) {
					action_11;
					goto tree_73;
				}
				else {
					if (condition_h) {
						if (condition_c) {
							action_11;
							goto tree_73;
						}
						else {
							action_14;
							goto tree_73;
						}
					}
					else {
						action_11;
						goto tree_73;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_k) {
						if (condition_i) {
							if (condition_d) {
								action_12;
								goto tree_4;
							}
							else {
								action_16;
								goto tree_4;
							}
						}
						else {
							if (condition_h) {
								if (condition_d) {
									if (condition_c) {
										action_12;
										goto tree_4;
									}
									else {
										action_15;
										goto tree_4;
									}
								}
								else {
									action_15;
									goto tree_4;
								}
							}
							else {
								action_12;
								goto tree_4;
							}
						}
					}
					else {
						if (condition_h) {
							action_9;
							goto tree_47;
						}
						else {
							if (condition_i) {
								action_11;
								goto tree_3;
							}
							else {
								action_6;
								goto tree_2;
							}
						}
					}
				}
				else {
					if (condition_h) {
						action_9;
						goto tree_82;
					}
					else {
						if (condition_i) {
							action_11;
							goto tree_63;
						}
						else {
							action_6;
							goto tree_61;
						}
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						action_11;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_d) {
								action_12;
								goto tree_4;
							}
							else {
								if (condition_i) {
									action_16;
									goto tree_4;
								}
								else {
									action_12;
									goto tree_4;
								}
							}
						}
						else {
							if (condition_i) {
								action_11;
								goto tree_3;
							}
							else {
								action_6;
								goto tree_2;
							}
						}
					}
				}
				else {
					action_6;
					goto tree_8;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_2: finish_condition(2);
		if (condition_o) {
			if (condition_j) {
				action_11;
				goto tree_73;
			}
			else {
				if (condition_p) {
					if (condition_k) {
						if (condition_d) {
							action_12;
							goto tree_4;
						}
						else {
							action_12;
							goto tree_4;
						}
					}
					else {
						action_6;
						goto tree_47;
					}
				}
				else {
					action_6;
					goto tree_82;
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						action_11;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_d) {
								action_12;
								goto tree_4;
							}
							else {
								action_12;
								goto tree_4;
							}
						}
						else {
							action_6;
							goto tree_47;
						}
					}
				}
				else {
					action_6;
					goto tree_8;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							action_5;
							goto tree_4;
						}
						else {
							action_2;
							goto tree_2;
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_3: finish_condition(3);
		if (condition_o) {
			if (condition_j) {
				if (condition_c) {
					if (condition_b) {
						action_6;
						goto tree_73;
					}
					else {
						action_11;
						goto tree_73;
					}
				}
				else {
					action_11;
					goto tree_73;
				}
			}
			else {
				if (condition_p) {
					if (condition_k) {
						if (condition_d) {
							if (condition_c) {
								if (condition_b) {
									action_6;
									goto tree_4;
								}
								else {
									action_12;
									goto tree_4;
								}
							}
							else {
								action_12;
								goto tree_4;
							}
						}
						else {
							action_12;
							goto tree_4;
						}
					}
					else {
						action_6;
						goto tree_47;
					}
				}
				else {
					action_6;
					goto tree_82;
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						if (condition_c) {
							if (condition_b) {
								action_6;
								goto tree_7;
							}
							else {
								action_11;
								goto tree_7;
							}
						}
						else {
							action_11;
							goto tree_7;
						}
					}
					else {
						if (condition_k) {
							if (condition_d) {
								if (condition_c) {
									if (condition_b) {
										action_6;
										goto tree_4;
									}
									else {
										action_12;
										goto tree_4;
									}
								}
								else {
									action_12;
									goto tree_4;
								}
							}
							else {
								action_12;
								goto tree_4;
							}
						}
						else {
							action_6;
							goto tree_47;
						}
					}
				}
				else {
					action_6;
					goto tree_8;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							action_5;
							goto tree_4;
						}
						else {
							action_2;
							goto tree_2;
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_4: finish_condition(4);
		if (condition_o) {
			if (condition_j) {
				action_6;
				goto tree_73;
			}
			else {
				if (condition_p) {
					if (condition_k) {
						if (condition_d) {
							action_6;
							goto tree_4;
						}
						else {
							action_12;
							goto tree_4;
						}
					}
					else {
						action_6;
						goto tree_47;
					}
				}
				else {
					action_6;
					goto tree_82;
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						action_6;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_d) {
								action_6;
								goto tree_4;
							}
							else {
								action_12;
								goto tree_4;
							}
						}
						else {
							action_6;
							goto tree_47;
						}
					}
				}
				else {
					action_6;
					goto tree_8;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_d) {
								action_5;
								goto tree_4;
							}
							else {
								action_10;
								goto tree_4;
							}
						}
						else {
							action_4;
							goto tree_3;
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_7: finish_condition(7);
		if (condition_o) {
			if (condition_j) {
				if (condition_i) {
					action_6;
					goto tree_73;
				}
				else {
					if (condition_c) {
						action_6;
						goto tree_73;
					}
					else {
						action_11;
						goto tree_73;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_k) {
						if (condition_d) {
							if (condition_i) {
								action_6;
								goto tree_4;
							}
							else {
								if (condition_c) {
									action_6;
									goto tree_4;
								}
								else {
									action_12;
									goto tree_4;
								}
							}
						}
						else {
							action_12;
							goto tree_4;
						}
					}
					else {
						action_6;
						goto tree_47;
					}
				}
				else {
					action_6;
					goto tree_82;
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						if (condition_i) {
							action_6;
							goto tree_7;
						}
						else {
							if (condition_c) {
								action_6;
								goto tree_7;
							}
							else {
								action_11;
								goto tree_7;
							}
						}
					}
					else {
						if (condition_k) {
							if (condition_d) {
								if (condition_i) {
									action_6;
									goto tree_4;
								}
								else {
									if (condition_c) {
										action_6;
										goto tree_4;
									}
									else {
										action_12;
										goto tree_4;
									}
								}
							}
							else {
								action_12;
								goto tree_4;
							}
						}
						else {
							action_6;
							goto tree_47;
						}
					}
				}
				else {
					action_6;
					goto tree_8;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_8: finish_condition(8);
		if (condition_o) {
			if (condition_r) {
				if (condition_j) {
					if (condition_i) {
						action_11;
						goto tree_73;
					}
					else {
						if (condition_h) {
							if (condition_c) {
								action_11;
								goto tree_73;
							}
							else {
								action_14;
								goto tree_73;
							}
						}
						else {
							action_11;
							goto tree_73;
						}
					}
				}
				else {
					if (condition_p) {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_12;
									goto tree_4;
								}
								else {
									action_16;
									goto tree_4;
								}
							}
							else {
								if (condition_h) {
									if (condition_d) {
										if (condition_c) {
											action_12;
											goto tree_4;
										}
										else {
											action_15;
											goto tree_4;
										}
									}
									else {
										action_15;
										goto tree_4;
									}
								}
								else {
									action_12;
									goto tree_4;
								}
							}
						}
						else {
							if (condition_h) {
								action_9;
								goto tree_47;
							}
							else {
								if (condition_i) {
									action_11;
									goto tree_3;
								}
								else {
									action_6;
									goto tree_2;
								}
							}
						}
					}
					else {
						if (condition_h) {
							action_9;
							goto tree_82;
						}
						else {
							if (condition_i) {
								action_11;
								goto tree_63;
							}
							else {
								action_6;
								goto tree_61;
							}
						}
					}
				}
			}
			else {
				if (condition_j) {
					if (condition_i) {
						action_4;
						goto tree_73;
					}
					else {
						if (condition_h) {
							if (condition_c) {
								action_4;
								goto tree_73;
							}
							else {
								action_7;
								goto tree_73;
							}
						}
						else {
							action_4;
							goto tree_73;
						}
					}
				}
				else {
					if (condition_p) {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								if (condition_h) {
									if (condition_d) {
										if (condition_c) {
											action_5;
											goto tree_4;
										}
										else {
											action_8;
											goto tree_4;
										}
									}
									else {
										action_8;
										goto tree_4;
									}
								}
								else {
									action_5;
									goto tree_4;
								}
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								if (condition_h) {
									action_3;
									goto tree_2;
								}
								else {
									action_2;
									goto tree_2;
								}
							}
						}
					}
					else {
						if (condition_i) {
							action_4;
							goto tree_63;
						}
						else {
							if (condition_h) {
								action_3;
								goto tree_61;
							}
							else {
								action_2;
								goto tree_61;
							}
						}
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_r) {
						if (condition_j) {
							action_11;
							goto tree_7;
						}
						else {
							if (condition_k) {
								if (condition_d) {
									action_12;
									goto tree_4;
								}
								else {
									if (condition_i) {
										action_16;
										goto tree_4;
									}
									else {
										action_12;
										goto tree_4;
									}
								}
							}
							else {
								if (condition_i) {
									action_11;
									goto tree_3;
								}
								else {
									action_6;
									goto tree_2;
								}
							}
						}
					}
					else {
						if (condition_j) {
							action_4;
							goto tree_7;
						}
						else {
							if (condition_k) {
								if (condition_i) {
									if (condition_d) {
										action_5;
										goto tree_4;
									}
									else {
										action_10;
										goto tree_4;
									}
								}
								else {
									action_5;
									goto tree_4;
								}
							}
							else {
								if (condition_i) {
									action_4;
									goto tree_3;
								}
								else {
									action_2;
									goto tree_2;
								}
							}
						}
					}
				}
				else {
					if (condition_r) {
						action_6;
						goto tree_8;
					}
					else {
						action_2;
						goto tree_8;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_47: finish_condition(47);
		if (condition_o) {
			if (condition_j) {
				if (condition_c) {
					if (condition_g) {
						if (condition_b) {
							action_6;
							goto tree_73;
						}
						else {
							action_11;
							goto tree_73;
						}
					}
					else {
						action_11;
						goto tree_73;
					}
				}
				else {
					action_11;
					goto tree_73;
				}
			}
			else {
				if (condition_p) {
					if (condition_k) {
						if (condition_d) {
							if (condition_c) {
								if (condition_g) {
									if (condition_b) {
										action_6;
										goto tree_4;
									}
									else {
										action_12;
										goto tree_4;
									}
								}
								else {
									action_12;
									goto tree_4;
								}
							}
							else {
								action_12;
								goto tree_4;
							}
						}
						else {
							action_12;
							goto tree_4;
						}
					}
					else {
						action_6;
						goto tree_47;
					}
				}
				else {
					action_6;
					goto tree_82;
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						if (condition_c) {
							if (condition_g) {
								if (condition_b) {
									action_6;
									goto tree_7;
								}
								else {
									action_11;
									goto tree_7;
								}
							}
							else {
								action_11;
								goto tree_7;
							}
						}
						else {
							action_11;
							goto tree_7;
						}
					}
					else {
						if (condition_k) {
							if (condition_d) {
								if (condition_c) {
									if (condition_g) {
										if (condition_b) {
											action_6;
											goto tree_4;
										}
										else {
											action_12;
											goto tree_4;
										}
									}
									else {
										action_12;
										goto tree_4;
									}
								}
								else {
									action_12;
									goto tree_4;
								}
							}
							else {
								action_12;
								goto tree_4;
							}
						}
						else {
							action_6;
							goto tree_47;
						}
					}
				}
				else {
					action_6;
					goto tree_8;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							action_5;
							goto tree_4;
						}
						else {
							action_2;
							goto tree_2;
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_61: finish_condition(61);
		if (condition_o) {
			if (condition_r) {
				if (condition_j) {
					action_11;
					goto tree_73;
				}
				else {
					if (condition_p) {
						if (condition_k) {
							if (condition_d) {
								action_12;
								goto tree_4;
							}
							else {
								if (condition_i) {
									action_16;
									goto tree_4;
								}
								else {
									action_12;
									goto tree_4;
								}
							}
						}
						else {
							if (condition_i) {
								action_11;
								goto tree_3;
							}
							else {
								action_6;
								goto tree_2;
							}
						}
					}
					else {
						if (condition_i) {
							action_11;
							goto tree_63;
						}
						else {
							action_6;
							goto tree_61;
						}
					}
				}
			}
			else {
				if (condition_j) {
					action_4;
					goto tree_73;
				}
				else {
					if (condition_p) {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
					else {
						if (condition_i) {
							action_4;
							goto tree_63;
						}
						else {
							action_2;
							goto tree_61;
						}
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_r) {
						if (condition_j) {
							action_11;
							goto tree_7;
						}
						else {
							if (condition_k) {
								if (condition_d) {
									action_12;
									goto tree_4;
								}
								else {
									if (condition_i) {
										action_16;
										goto tree_4;
									}
									else {
										action_12;
										goto tree_4;
									}
								}
							}
							else {
								if (condition_i) {
									action_11;
									goto tree_3;
								}
								else {
									action_6;
									goto tree_2;
								}
							}
						}
					}
					else {
						if (condition_j) {
							action_4;
							goto tree_7;
						}
						else {
							if (condition_k) {
								if (condition_i) {
									if (condition_d) {
										action_5;
										goto tree_4;
									}
									else {
										action_10;
										goto tree_4;
									}
								}
								else {
									action_5;
									goto tree_4;
								}
							}
							else {
								if (condition_i) {
									action_4;
									goto tree_3;
								}
								else {
									action_2;
									goto tree_2;
								}
							}
						}
					}
				}
				else {
					if (condition_r) {
						action_6;
						goto tree_8;
					}
					else {
						action_2;
						goto tree_8;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_63: finish_condition(63);
		if (condition_o) {
			if (condition_r) {
				if (condition_j) {
					if (condition_b) {
						if (condition_i) {
							action_6;
							goto tree_73;
						}
						else {
							if (condition_c) {
								action_6;
								goto tree_73;
							}
							else {
								action_11;
								goto tree_73;
							}
						}
					}
					else {
						action_11;
						goto tree_73;
					}
				}
				else {
					if (condition_p) {
						if (condition_k) {
							if (condition_d) {
								if (condition_b) {
									if (condition_i) {
										action_6;
										goto tree_4;
									}
									else {
										if (condition_c) {
											action_6;
											goto tree_4;
										}
										else {
											action_12;
											goto tree_4;
										}
									}
								}
								else {
									action_12;
									goto tree_4;
								}
							}
							else {
								if (condition_i) {
									if (condition_b) {
										action_12;
										goto tree_4;
									}
									else {
										action_16;
										goto tree_4;
									}
								}
								else {
									action_12;
									goto tree_4;
								}
							}
						}
						else {
							if (condition_i) {
								if (condition_b) {
									action_6;
									goto tree_3;
								}
								else {
									action_11;
									goto tree_3;
								}
							}
							else {
								action_6;
								goto tree_2;
							}
						}
					}
					else {
						if (condition_i) {
							if (condition_b) {
								action_6;
								goto tree_63;
							}
							else {
								action_11;
								goto tree_63;
							}
						}
						else {
							action_6;
							goto tree_61;
						}
					}
				}
			}
			else {
				if (condition_j) {
					action_4;
					goto tree_73;
				}
				else {
					if (condition_p) {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
					else {
						if (condition_i) {
							action_4;
							goto tree_63;
						}
						else {
							action_2;
							goto tree_61;
						}
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_r) {
						if (condition_j) {
							if (condition_b) {
								if (condition_i) {
									action_6;
									goto tree_7;
								}
								else {
									if (condition_c) {
										action_6;
										goto tree_7;
									}
									else {
										action_11;
										goto tree_7;
									}
								}
							}
							else {
								action_11;
								goto tree_7;
							}
						}
						else {
							if (condition_k) {
								if (condition_d) {
									if (condition_b) {
										if (condition_i) {
											action_6;
											goto tree_4;
										}
										else {
											if (condition_c) {
												action_6;
												goto tree_4;
											}
											else {
												action_12;
												goto tree_4;
											}
										}
									}
									else {
										action_12;
										goto tree_4;
									}
								}
								else {
									if (condition_i) {
										if (condition_b) {
											action_12;
											goto tree_4;
										}
										else {
											action_16;
											goto tree_4;
										}
									}
									else {
										action_12;
										goto tree_4;
									}
								}
							}
							else {
								if (condition_i) {
									if (condition_b) {
										action_6;
										goto tree_3;
									}
									else {
										action_11;
										goto tree_3;
									}
								}
								else {
									action_6;
									goto tree_2;
								}
							}
						}
					}
					else {
						if (condition_j) {
							action_4;
							goto tree_7;
						}
						else {
							if (condition_k) {
								if (condition_i) {
									if (condition_d) {
										action_5;
										goto tree_4;
									}
									else {
										action_10;
										goto tree_4;
									}
								}
								else {
									action_5;
									goto tree_4;
								}
							}
							else {
								if (condition_i) {
									action_4;
									goto tree_3;
								}
								else {
									action_2;
									goto tree_2;
								}
							}
						}
					}
				}
				else {
					if (condition_r) {
						action_6;
						goto tree_8;
					}
					else {
						action_2;
						goto tree_8;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_73: finish_condition(73);
		if (condition_o) {
			if (condition_n) {
				if (condition_j) {
					if (condition_i) {
						action_6;
						goto tree_73;
					}
					else {
						if (condition_c) {
							action_6;
							goto tree_73;
						}
						else {
							action_11;
							goto tree_73;
						}
					}
				}
				else {
					if (condition_p) {
						if (condition_k) {
							if (condition_d) {
								if (condition_i) {
									action_6;
									goto tree_4;
								}
								else {
									if (condition_c) {
										action_6;
										goto tree_4;
									}
									else {
										action_12;
										goto tree_4;
									}
								}
							}
							else {
								action_12;
								goto tree_4;
							}
						}
						else {
							action_6;
							goto tree_47;
						}
					}
					else {
						action_6;
						goto tree_82;
					}
				}
			}
			else {
				if (condition_r) {
					if (condition_j) {
						if (condition_i) {
							action_6;
							goto tree_73;
						}
						else {
							if (condition_c) {
								action_6;
								goto tree_73;
							}
							else {
								action_11;
								goto tree_73;
							}
						}
					}
					else {
						if (condition_p) {
							if (condition_k) {
								if (condition_d) {
									if (condition_i) {
										action_6;
										goto tree_4;
									}
									else {
										if (condition_c) {
											action_6;
											goto tree_4;
										}
										else {
											action_12;
											goto tree_4;
										}
									}
								}
								else {
									action_12;
									goto tree_4;
								}
							}
							else {
								action_6;
								goto tree_47;
							}
						}
						else {
							action_6;
							goto tree_82;
						}
					}
				}
				else {
					if (condition_j) {
						if (condition_i) {
							action_4;
							goto tree_73;
						}
						else {
							if (condition_c) {
								action_4;
								goto tree_73;
							}
							else {
								action_7;
								goto tree_73;
							}
						}
					}
					else {
						if (condition_p) {
							if (condition_k) {
								if (condition_i) {
									if (condition_d) {
										action_5;
										goto tree_4;
									}
									else {
										action_10;
										goto tree_4;
									}
								}
								else {
									if (condition_d) {
										if (condition_c) {
											action_5;
											goto tree_4;
										}
										else {
											action_8;
											goto tree_4;
										}
									}
									else {
										action_8;
										goto tree_4;
									}
								}
							}
							else {
								if (condition_i) {
									action_4;
									goto tree_3;
								}
								else {
									action_3;
									goto tree_2;
								}
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_63;
							}
							else {
								action_3;
								goto tree_61;
							}
						}
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_n) {
						if (condition_j) {
							if (condition_i) {
								action_6;
								goto tree_7;
							}
							else {
								if (condition_c) {
									action_6;
									goto tree_7;
								}
								else {
									action_11;
									goto tree_7;
								}
							}
						}
						else {
							if (condition_k) {
								if (condition_d) {
									if (condition_i) {
										action_6;
										goto tree_4;
									}
									else {
										if (condition_c) {
											action_6;
											goto tree_4;
										}
										else {
											action_12;
											goto tree_4;
										}
									}
								}
								else {
									action_12;
									goto tree_4;
								}
							}
							else {
								action_6;
								goto tree_47;
							}
						}
					}
					else {
						if (condition_r) {
							if (condition_j) {
								if (condition_i) {
									action_6;
									goto tree_7;
								}
								else {
									if (condition_c) {
										action_6;
										goto tree_7;
									}
									else {
										action_11;
										goto tree_7;
									}
								}
							}
							else {
								if (condition_k) {
									if (condition_d) {
										if (condition_i) {
											action_6;
											goto tree_4;
										}
										else {
											if (condition_c) {
												action_6;
												goto tree_4;
											}
											else {
												action_12;
												goto tree_4;
											}
										}
									}
									else {
										action_12;
										goto tree_4;
									}
								}
								else {
									if (condition_i) {
										action_6;
										goto tree_3;
									}
									else {
										action_6;
										goto tree_2;
									}
								}
							}
						}
						else {
							if (condition_j) {
								action_4;
								goto tree_7;
							}
							else {
								if (condition_k) {
									if (condition_i) {
										if (condition_d) {
											action_5;
											goto tree_4;
										}
										else {
											action_10;
											goto tree_4;
										}
									}
									else {
										action_5;
										goto tree_4;
									}
								}
								else {
									if (condition_i) {
										action_4;
										goto tree_3;
									}
									else {
										action_2;
										goto tree_2;
									}
								}
							}
						}
					}
				}
				else {
					if (condition_r) {
						action_6;
						goto tree_8;
					}
					else {
						if (condition_n) {
							action_6;
							goto tree_8;
						}
						else {
							action_2;
							goto tree_8;
						}
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_82: finish_condition(82);
		if (condition_o) {
			if (condition_r) {
				if (condition_j) {
					if (condition_g) {
						if (condition_b) {
							if (condition_i) {
								action_6;
								goto tree_73;
							}
							else {
								if (condition_c) {
									action_6;
									goto tree_73;
								}
								else {
									action_11;
									goto tree_73;
								}
							}
						}
						else {
							action_11;
							goto tree_73;
						}
					}
					else {
						action_11;
						goto tree_73;
					}
				}
				else {
					if (condition_p) {
						if (condition_k) {
							if (condition_d) {
								if (condition_g) {
									if (condition_b) {
										if (condition_i) {
											action_6;
											goto tree_4;
										}
										else {
											if (condition_c) {
												action_6;
												goto tree_4;
											}
											else {
												action_12;
												goto tree_4;
											}
										}
									}
									else {
										action_12;
										goto tree_4;
									}
								}
								else {
									action_12;
									goto tree_4;
								}
							}
							else {
								if (condition_i) {
									if (condition_g) {
										if (condition_b) {
											action_12;
											goto tree_4;
										}
										else {
											action_16;
											goto tree_4;
										}
									}
									else {
										action_16;
										goto tree_4;
									}
								}
								else {
									action_12;
									goto tree_4;
								}
							}
						}
						else {
							if (condition_i) {
								if (condition_g) {
									if (condition_b) {
										action_6;
										goto tree_3;
									}
									else {
										action_11;
										goto tree_3;
									}
								}
								else {
									action_11;
									goto tree_3;
								}
							}
							else {
								action_6;
								goto tree_2;
							}
						}
					}
					else {
						if (condition_i) {
							if (condition_g) {
								if (condition_b) {
									action_6;
									goto tree_63;
								}
								else {
									action_11;
									goto tree_63;
								}
							}
							else {
								action_11;
								goto tree_63;
							}
						}
						else {
							action_6;
							goto tree_61;
						}
					}
				}
			}
			else {
				if (condition_j) {
					action_4;
					goto tree_73;
				}
				else {
					if (condition_p) {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
					else {
						if (condition_i) {
							action_4;
							goto tree_63;
						}
						else {
							action_2;
							goto tree_61;
						}
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_r) {
						if (condition_j) {
							if (condition_g) {
								if (condition_b) {
									if (condition_i) {
										action_6;
										goto tree_7;
									}
									else {
										if (condition_c) {
											action_6;
											goto tree_7;
										}
										else {
											action_11;
											goto tree_7;
										}
									}
								}
								else {
									action_11;
									goto tree_7;
								}
							}
							else {
								action_11;
								goto tree_7;
							}
						}
						else {
							if (condition_k) {
								if (condition_d) {
									if (condition_g) {
										if (condition_b) {
											if (condition_i) {
												action_6;
												goto tree_4;
											}
											else {
												if (condition_c) {
													action_6;
													goto tree_4;
												}
												else {
													action_12;
													goto tree_4;
												}
											}
										}
										else {
											action_12;
											goto tree_4;
										}
									}
									else {
										action_12;
										goto tree_4;
									}
								}
								else {
									if (condition_i) {
										if (condition_g) {
											if (condition_b) {
												action_12;
												goto tree_4;
											}
											else {
												action_16;
												goto tree_4;
											}
										}
										else {
											action_16;
											goto tree_4;
										}
									}
									else {
										action_12;
										goto tree_4;
									}
								}
							}
							else {
								if (condition_i) {
									if (condition_g) {
										if (condition_b) {
											action_6;
											goto tree_3;
										}
										else {
											action_11;
											goto tree_3;
										}
									}
									else {
										action_11;
										goto tree_3;
									}
								}
								else {
									action_6;
									goto tree_2;
								}
							}
						}
					}
					else {
						if (condition_j) {
							action_4;
							goto tree_7;
						}
						else {
							if (condition_k) {
								if (condition_i) {
									if (condition_d) {
										action_5;
										goto tree_4;
									}
									else {
										action_10;
										goto tree_4;
									}
								}
								else {
									action_5;
									goto tree_4;
								}
							}
							else {
								if (condition_i) {
									action_4;
									goto tree_3;
								}
								else {
									action_2;
									goto tree_2;
								}
							}
						}
					}
				}
				else {
					if (condition_r) {
						action_6;
						goto tree_8;
					}
					else {
						action_2;
						goto tree_8;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	break_0:
		if (condition_o) {
			if (condition_j) {
				if (condition_i) {
					action_4;
				}
				else {
					if (condition_h) {
						if (condition_c) {
							action_4;
						}
						else {
							action_7;
						}
					}
					else {
						action_4;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_i) {
						action_4;
					}
					else {
						if (condition_h) {
							action_3;
						}
						else {
							action_2;
						}
					}
				}
				else {
					if (condition_i) {
						action_4;
					}
					else {
						if (condition_h) {
							action_3;
						}
						else {
							action_2;
						}
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						action_4;
					}
					else {
						if (condition_i) {
							action_4;
						}
						else {
							action_2;
						}
					}
				}
				else {
					action_2;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
					}
					else {
						if (condition_i) {
							action_4;
						}
						else {
							action_2;
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
					}
					else {
						action_1;
					}
				}
			}
		}
		continue;
	break_1:
		if (condition_o) {
			if (condition_j) {
				if (condition_i) {
					action_11;
				}
				else {
					if (condition_h) {
						if (condition_c) {
							action_11;
						}
						else {
							action_14;
						}
					}
					else {
						action_11;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_h) {
						action_9;
					}
					else {
						if (condition_i) {
							action_11;
						}
						else {
							action_6;
						}
					}
				}
				else {
					if (condition_h) {
						action_9;
					}
					else {
						if (condition_i) {
							action_11;
						}
						else {
							action_6;
						}
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						action_11;
					}
					else {
						if (condition_i) {
							action_11;
						}
						else {
							action_6;
						}
					}
				}
				else {
					action_6;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
					}
					else {
						if (condition_i) {
							action_4;
						}
						else {
							action_2;
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
					}
					else {
						action_1;
					}
				}
			}
		}
		continue;
	break_2:
		if (condition_o) {
			if (condition_j) {
				action_11;
			}
			else {
				if (condition_p) {
					action_6;
				}
				else {
					action_6;
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						action_11;
					}
					else {
						action_6;
					}
				}
				else {
					action_6;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
					}
					else {
						action_2;
					}
				}
				else {
					if (condition_t) {
						action_2;
					}
					else {
						action_1;
					}
				}
			}
		}
		continue;
	break_3:
		if (condition_o) {
			if (condition_j) {
				if (condition_c) {
					if (condition_b) {
						action_6;
					}
					else {
						action_11;
					}
				}
				else {
					action_11;
				}
			}
			else {
				if (condition_p) {
					action_6;
				}
				else {
					action_6;
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						if (condition_c) {
							if (condition_b) {
								action_6;
							}
							else {
								action_11;
							}
						}
						else {
							action_11;
						}
					}
					else {
						action_6;
					}
				}
				else {
					action_6;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
					}
					else {
						action_2;
					}
				}
				else {
					if (condition_t) {
						action_2;
					}
					else {
						action_1;
					}
				}
			}
		}
		continue;
	break_4:
		if (condition_o) {
			if (condition_j) {
				action_6;
			}
			else {
				if (condition_p) {
					action_6;
				}
				else {
					action_6;
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						action_6;
					}
					else {
						action_6;
					}
				}
				else {
					action_6;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
					}
					else {
						action_4;
					}
				}
				else {
					if (condition_t) {
						action_2;
					}
					else {
						action_1;
					}
				}
			}
		}
		continue;
	break_7:
		if (condition_o) {
			if (condition_j) {
				if (condition_i) {
					action_6;
				}
				else {
					if (condition_c) {
						action_6;
					}
					else {
						action_11;
					}
				}
			}
			else {
				if (condition_p) {
					action_6;
				}
				else {
					action_6;
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						if (condition_i) {
							action_6;
						}
						else {
							if (condition_c) {
								action_6;
							}
							else {
								action_11;
							}
						}
					}
					else {
						action_6;
					}
				}
				else {
					action_6;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
					}
					else {
						if (condition_i) {
							action_4;
						}
						else {
							action_2;
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
					}
					else {
						action_1;
					}
				}
			}
		}
		continue;
	break_8:
		if (condition_o) {
			if (condition_r) {
				if (condition_j) {
					if (condition_i) {
						action_11;
					}
					else {
						if (condition_h) {
							if (condition_c) {
								action_11;
							}
							else {
								action_14;
							}
						}
						else {
							action_11;
						}
					}
				}
				else {
					if (condition_p) {
						if (condition_h) {
							action_9;
						}
						else {
							if (condition_i) {
								action_11;
							}
							else {
								action_6;
							}
						}
					}
					else {
						if (condition_h) {
							action_9;
						}
						else {
							if (condition_i) {
								action_11;
							}
							else {
								action_6;
							}
						}
					}
				}
			}
			else {
				if (condition_j) {
					if (condition_i) {
						action_4;
					}
					else {
						if (condition_h) {
							if (condition_c) {
								action_4;
							}
							else {
								action_7;
							}
						}
						else {
							action_4;
						}
					}
				}
				else {
					if (condition_p) {
						if (condition_i) {
							action_4;
						}
						else {
							if (condition_h) {
								action_3;
							}
							else {
								action_2;
							}
						}
					}
					else {
						if (condition_i) {
							action_4;
						}
						else {
							if (condition_h) {
								action_3;
							}
							else {
								action_2;
							}
						}
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_r) {
						if (condition_j) {
							action_11;
						}
						else {
							if (condition_i) {
								action_11;
							}
							else {
								action_6;
							}
						}
					}
					else {
						if (condition_j) {
							action_4;
						}
						else {
							if (condition_i) {
								action_4;
							}
							else {
								action_2;
							}
						}
					}
				}
				else {
					if (condition_r) {
						action_6;
					}
					else {
						action_2;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
					}
					else {
						if (condition_i) {
							action_4;
						}
						else {
							action_2;
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
					}
					else {
						action_1;
					}
				}
			}
		}
		continue;
	break_47:
		if (condition_o) {
			if (condition_j) {
				if (condition_c) {
					if (condition_g) {
						if (condition_b) {
							action_6;
						}
						else {
							action_11;
						}
					}
					else {
						action_11;
					}
				}
				else {
					action_11;
				}
			}
			else {
				if (condition_p) {
					action_6;
				}
				else {
					action_6;
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						if (condition_c) {
							if (condition_g) {
								if (condition_b) {
									action_6;
								}
								else {
									action_11;
								}
							}
							else {
								action_11;
							}
						}
						else {
							action_11;
						}
					}
					else {
						action_6;
					}
				}
				else {
					action_6;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
					}
					else {
						action_2;
					}
				}
				else {
					if (condition_t) {
						action_2;
					}
					else {
						action_1;
					}
				}
			}
		}
		continue;
	break_61:
		if (condition_o) {
			if (condition_r) {
				if (condition_j) {
					action_11;
				}
				else {
					if (condition_p) {
						if (condition_i) {
							action_11;
						}
						else {
							action_6;
						}
					}
					else {
						if (condition_i) {
							action_11;
						}
						else {
							action_6;
						}
					}
				}
			}
			else {
				if (condition_j) {
					action_4;
				}
				else {
					if (condition_p) {
						if (condition_i) {
							action_4;
						}
						else {
							action_2;
						}
					}
					else {
						if (condition_i) {
							action_4;
						}
						else {
							action_2;
						}
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_r) {
						if (condition_j) {
							action_11;
						}
						else {
							if (condition_i) {
								action_11;
							}
							else {
								action_6;
							}
						}
					}
					else {
						if (condition_j) {
							action_4;
						}
						else {
							if (condition_i) {
								action_4;
							}
							else {
								action_2;
							}
						}
					}
				}
				else {
					if (condition_r) {
						action_6;
					}
					else {
						action_2;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
					}
					else {
						if (condition_i) {
							action_4;
						}
						else {
							action_2;
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
					}
					else {
						action_1;
					}
				}
			}
		}
		continue;
	break_63:
		if (condition_o) {
			if (condition_r) {
				if (condition_j) {
					if (condition_b) {
						if (condition_i) {
							action_6;
						}
						else {
							if (condition_c) {
								action_6;
							}
							else {
								action_11;
							}
						}
					}
					else {
						action_11;
					}
				}
				else {
					if (condition_p) {
						if (condition_i) {
							if (condition_b) {
								action_6;
							}
							else {
								action_11;
							}
						}
						else {
							action_6;
						}
					}
					else {
						if (condition_i) {
							if (condition_b) {
								action_6;
							}
							else {
								action_11;
							}
						}
						else {
							action_6;
						}
					}
				}
			}
			else {
				if (condition_j) {
					action_4;
				}
				else {
					if (condition_p) {
						if (condition_i) {
							action_4;
						}
						else {
							action_2;
						}
					}
					else {
						if (condition_i) {
							action_4;
						}
						else {
							action_2;
						}
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_r) {
						if (condition_j) {
							if (condition_b) {
								if (condition_i) {
									action_6;
								}
								else {
									if (condition_c) {
										action_6;
									}
									else {
										action_11;
									}
								}
							}
							else {
								action_11;
							}
						}
						else {
							if (condition_i) {
								if (condition_b) {
									action_6;
								}
								else {
									action_11;
								}
							}
							else {
								action_6;
							}
						}
					}
					else {
						if (condition_j) {
							action_4;
						}
						else {
							if (condition_i) {
								action_4;
							}
							else {
								action_2;
							}
						}
					}
				}
				else {
					if (condition_r) {
						action_6;
					}
					else {
						action_2;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
					}
					else {
						if (condition_i) {
							action_4;
						}
						else {
							action_2;
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
					}
					else {
						action_1;
					}
				}
			}
		}
		continue;
	break_73:
		if (condition_o) {
			if (condition_n) {
				if (condition_j) {
					if (condition_i) {
						action_6;
					}
					else {
						if (condition_c) {
							action_6;
						}
						else {
							action_11;
						}
					}
				}
				else {
					if (condition_p) {
						action_6;
					}
					else {
						action_6;
					}
				}
			}
			else {
				if (condition_r) {
					if (condition_j) {
						if (condition_i) {
							action_6;
						}
						else {
							if (condition_c) {
								action_6;
							}
							else {
								action_11;
							}
						}
					}
					else {
						if (condition_p) {
							action_6;
						}
						else {
							action_6;
						}
					}
				}
				else {
					if (condition_j) {
						if (condition_i) {
							action_4;
						}
						else {
							if (condition_c) {
								action_4;
							}
							else {
								action_7;
							}
						}
					}
					else {
						if (condition_p) {
							if (condition_i) {
								action_4;
							}
							else {
								action_3;
							}
						}
						else {
							if (condition_i) {
								action_4;
							}
							else {
								action_3;
							}
						}
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_n) {
						if (condition_j) {
							if (condition_i) {
								action_6;
							}
							else {
								if (condition_c) {
									action_6;
								}
								else {
									action_11;
								}
							}
						}
						else {
							action_6;
						}
					}
					else {
						if (condition_r) {
							if (condition_j) {
								if (condition_i) {
									action_6;
								}
								else {
									if (condition_c) {
										action_6;
									}
									else {
										action_11;
									}
								}
							}
							else {
								if (condition_i) {
									action_6;
								}
								else {
									action_6;
								}
							}
						}
						else {
							if (condition_j) {
								action_4;
							}
							else {
								if (condition_i) {
									action_4;
								}
								else {
									action_2;
								}
							}
						}
					}
				}
				else {
					if (condition_r) {
						action_6;
					}
					else {
						if (condition_n) {
							action_6;
						}
						else {
							action_2;
						}
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
					}
					else {
						if (condition_i) {
							action_4;
						}
						else {
							action_2;
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
					}
					else {
						action_1;
					}
				}
			}
		}
		continue;
	break_82:
		if (condition_o) {
			if (condition_r) {
				if (condition_j) {
					if (condition_g) {
						if (condition_b) {
							if (condition_i) {
								action_6;
							}
							else {
								if (condition_c) {
									action_6;
								}
								else {
									action_11;
								}
							}
						}
						else {
							action_11;
						}
					}
					else {
						action_11;
					}
				}
				else {
					if (condition_p) {
						if (condition_i) {
							if (condition_g) {
								if (condition_b) {
									action_6;
								}
								else {
									action_11;
								}
							}
							else {
								action_11;
							}
						}
						else {
							action_6;
						}
					}
					else {
						if (condition_i) {
							if (condition_g) {
								if (condition_b) {
									action_6;
								}
								else {
									action_11;
								}
							}
							else {
								action_11;
							}
						}
						else {
							action_6;
						}
					}
				}
			}
			else {
				if (condition_j) {
					action_4;
				}
				else {
					if (condition_p) {
						if (condition_i) {
							action_4;
						}
						else {
							action_2;
						}
					}
					else {
						if (condition_i) {
							action_4;
						}
						else {
							action_2;
						}
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_r) {
						if (condition_j) {
							if (condition_g) {
								if (condition_b) {
									if (condition_i) {
										action_6;
									}
									else {
										if (condition_c) {
											action_6;
										}
										else {
											action_11;
										}
									}
								}
								else {
									action_11;
								}
							}
							else {
								action_11;
							}
						}
						else {
							if (condition_i) {
								if (condition_g) {
									if (condition_b) {
										action_6;
									}
									else {
										action_11;
									}
								}
								else {
									action_11;
								}
							}
							else {
								action_6;
							}
						}
					}
					else {
						if (condition_j) {
							action_4;
						}
						else {
							if (condition_i) {
								action_4;
							}
							else {
								action_2;
							}
						}
					}
				}
				else {
					if (condition_r) {
						action_6;
					}
					else {
						action_2;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
					}
					else {
						if (condition_i) {
							action_4;
						}
						else {
							action_2;
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
					}
					else {
						action_1;
					}
				}
			}
		}
		continue;
	}

	for (int r = h - 1; r < h; r += 2) {
		// Get row pointers
		const uchar* const img_row = img.ptr<uchar>(r);
		uint* const imgLabels_row = imgLabels.ptr<uint>(r);
		uint* const imgLabels_row_prev_prev = (uint *)(((char *)imgLabels_row) - imgLabels.step.p[0] - imgLabels.step.p[0]);

		int c = -2;
		lr_finish_condition(0);
		if (condition_o) {
			if (condition_j) {
				if (condition_i) {
					action_4;
					goto lr_tree_19;
				}
				else {
					action_4;
					goto lr_tree_19;
				}
			}
			else {
				if (condition_p) {
					if (condition_k) {
						if (condition_i) {
							if (condition_d) {
								action_5;
								goto lr_tree_3;
							}
							else {
								action_10;
								goto lr_tree_3;
							}
						}
						else {
							action_5;
							goto lr_tree_3;
						}
					}
					else {
						if (condition_i) {
							action_4;
							goto lr_tree_2;
						}
						else {
							action_2;
							goto lr_tree_1;
						}
					}
				}
				else {
					if (condition_i) {
						action_4;
						goto lr_tree_7;
					}
					else {
						action_2;
						goto lr_tree_7;
					}
				}
			}
		}
		else {
			if (condition_p) {
				if (condition_j) {
					action_4;
					goto lr_tree_6;
				}
				else {
					if (condition_k) {
						if (condition_i) {
							if (condition_d) {
								action_5;
								goto lr_tree_3;
							}
							else {
								action_10;
								goto lr_tree_3;
							}
						}
						else {
							action_5;
							goto lr_tree_3;
						}
					}
					else {
						if (condition_i) {
							action_4;
							goto lr_tree_2;
						}
						else {
							action_2;
							goto lr_tree_1;
						}
					}
				}
			}
			else {
				action_1;
				goto lr_tree_0;
			}
		}
	lr_tree_0: lr_finish_condition(0);
		if (condition_o) {
			if (condition_j) {
				if (condition_i) {
					action_4;
					goto lr_tree_19;
				}
				else {
					if (condition_h) {
						if (condition_c) {
							action_4;
							goto lr_tree_19;
						}
						else {
							action_7;
							goto lr_tree_19;
						}
					}
					else {
						action_4;
						goto lr_tree_19;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_k) {
						if (condition_i) {
							if (condition_d) {
								action_5;
								goto lr_tree_3;
							}
							else {
								action_10;
								goto lr_tree_3;
							}
						}
						else {
							if (condition_h) {
								if (condition_d) {
									if (condition_c) {
										action_5;
										goto lr_tree_3;
									}
									else {
										action_8;
										goto lr_tree_3;
									}
								}
								else {
									action_8;
									goto lr_tree_3;
								}
							}
							else {
								action_5;
								goto lr_tree_3;
							}
						}
					}
					else {
						if (condition_i) {
							action_4;
							goto lr_tree_2;
						}
						else {
							if (condition_h) {
								action_3;
								goto lr_tree_1;
							}
							else {
								action_2;
								goto lr_tree_1;
							}
						}
					}
				}
				else {
					if (condition_i) {
						action_4;
						goto lr_tree_7;
					}
					else {
						if (condition_h) {
							action_3;
							goto lr_tree_7;
						}
						else {
							action_2;
							goto lr_tree_7;
						}
					}
				}
			}
		}
		else {
			if (condition_p) {
				if (condition_j) {
					action_4;
					goto lr_tree_6;
				}
				else {
					if (condition_k) {
						if (condition_i) {
							if (condition_d) {
								action_5;
								goto lr_tree_3;
							}
							else {
								action_10;
								goto lr_tree_3;
							}
						}
						else {
							action_5;
							goto lr_tree_3;
						}
					}
					else {
						if (condition_i) {
							action_4;
							goto lr_tree_2;
						}
						else {
							action_2;
							goto lr_tree_1;
						}
					}
				}
			}
			else {
				action_1;
				goto lr_tree_0;
			}
		}
	lr_tree_1: lr_finish_condition(1);
		if (condition_o) {
			if (condition_j) {
				action_11;
				goto lr_tree_19;
			}
			else {
				if (condition_p) {
					if (condition_k) {
						if (condition_d) {
							action_12;
							goto lr_tree_3;
						}
						else {
							action_12;
							goto lr_tree_3;
						}
					}
					else {
						action_6;
						goto lr_tree_24;
					}
				}
				else {
					action_6;
					goto lr_tree_7;
				}
			}
		}
		else {
			if (condition_p) {
				if (condition_j) {
					action_4;
					goto lr_tree_6;
				}
				else {
					if (condition_k) {
						action_5;
						goto lr_tree_3;
					}
					else {
						action_2;
						goto lr_tree_1;
					}
				}
			}
			else {
				action_1;
				goto lr_tree_0;
			}
		}
	lr_tree_2: lr_finish_condition(2);
		if (condition_o) {
			if (condition_j) {
				if (condition_c) {
					if (condition_b) {
						action_6;
						goto lr_tree_19;
					}
					else {
						action_11;
						goto lr_tree_19;
					}
				}
				else {
					action_11;
					goto lr_tree_19;
				}
			}
			else {
				if (condition_p) {
					if (condition_k) {
						if (condition_d) {
							if (condition_c) {
								if (condition_b) {
									action_6;
									goto lr_tree_3;
								}
								else {
									action_12;
									goto lr_tree_3;
								}
							}
							else {
								action_12;
								goto lr_tree_3;
							}
						}
						else {
							action_12;
							goto lr_tree_3;
						}
					}
					else {
						action_6;
						goto lr_tree_24;
					}
				}
				else {
					action_6;
					goto lr_tree_7;
				}
			}
		}
		else {
			if (condition_p) {
				if (condition_j) {
					action_4;
					goto lr_tree_6;
				}
				else {
					if (condition_k) {
						action_5;
						goto lr_tree_3;
					}
					else {
						action_2;
						goto lr_tree_1;
					}
				}
			}
			else {
				action_1;
				goto lr_tree_0;
			}
		}
	lr_tree_3: lr_finish_condition(3);
		if (condition_o) {
			if (condition_j) {
				action_6;
				goto lr_tree_19;
			}
			else {
				if (condition_p) {
					if (condition_k) {
						if (condition_d) {
							action_6;
							goto lr_tree_3;
						}
						else {
							action_12;
							goto lr_tree_3;
						}
					}
					else {
						action_6;
						goto lr_tree_24;
					}
				}
				else {
					action_6;
					goto lr_tree_7;
				}
			}
		}
		else {
			if (condition_p) {
				if (condition_j) {
					action_4;
					goto lr_tree_6;
				}
				else {
					if (condition_k) {
						if (condition_d) {
							action_5;
							goto lr_tree_3;
						}
						else {
							action_10;
							goto lr_tree_3;
						}
					}
					else {
						action_4;
						goto lr_tree_2;
					}
				}
			}
			else {
				action_1;
				goto lr_tree_0;
			}
		}
	lr_tree_6: lr_finish_condition(6);
		if (condition_o) {
			if (condition_j) {
				if (condition_i) {
					action_6;
					goto lr_tree_19;
				}
				else {
					if (condition_c) {
						action_6;
						goto lr_tree_19;
					}
					else {
						action_11;
						goto lr_tree_19;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_k) {
						if (condition_d) {
							if (condition_i) {
								action_6;
								goto lr_tree_3;
							}
							else {
								if (condition_c) {
									action_6;
									goto lr_tree_3;
								}
								else {
									action_12;
									goto lr_tree_3;
								}
							}
						}
						else {
							action_12;
							goto lr_tree_3;
						}
					}
					else {
						action_6;
						goto lr_tree_24;
					}
				}
				else {
					action_6;
					goto lr_tree_7;
				}
			}
		}
		else {
			if (condition_p) {
				if (condition_j) {
					action_4;
					goto lr_tree_6;
				}
				else {
					if (condition_k) {
						if (condition_i) {
							if (condition_d) {
								action_5;
								goto lr_tree_3;
							}
							else {
								action_10;
								goto lr_tree_3;
							}
						}
						else {
							action_5;
							goto lr_tree_3;
						}
					}
					else {
						if (condition_i) {
							action_4;
							goto lr_tree_2;
						}
						else {
							action_2;
							goto lr_tree_1;
						}
					}
				}
			}
			else {
				action_1;
				goto lr_tree_0;
			}
		}
	lr_tree_7: lr_finish_condition(7);
		if (condition_o) {
			if (condition_j) {
				action_4;
				goto lr_tree_19;
			}
			else {
				if (condition_p) {
					if (condition_k) {
						if (condition_i) {
							if (condition_d) {
								action_5;
								goto lr_tree_3;
							}
							else {
								action_10;
								goto lr_tree_3;
							}
						}
						else {
							action_5;
							goto lr_tree_3;
						}
					}
					else {
						if (condition_i) {
							action_4;
							goto lr_tree_2;
						}
						else {
							action_2;
							goto lr_tree_1;
						}
					}
				}
				else {
					if (condition_i) {
						action_4;
						goto lr_tree_7;
					}
					else {
						action_2;
						goto lr_tree_7;
					}
				}
			}
		}
		else {
			if (condition_p) {
				if (condition_j) {
					action_4;
					goto lr_tree_6;
				}
				else {
					if (condition_k) {
						if (condition_i) {
							if (condition_d) {
								action_5;
								goto lr_tree_3;
							}
							else {
								action_10;
								goto lr_tree_3;
							}
						}
						else {
							action_5;
							goto lr_tree_3;
						}
					}
					else {
						if (condition_i) {
							action_4;
							goto lr_tree_2;
						}
						else {
							action_2;
							goto lr_tree_1;
						}
					}
				}
			}
			else {
				action_1;
				goto lr_tree_0;
			}
		}
	lr_tree_19: lr_finish_condition(19);
		if (condition_o) {
			if (condition_n) {
				if (condition_j) {
					if (condition_i) {
						action_6;
						goto lr_tree_19;
					}
					else {
						if (condition_c) {
							action_6;
							goto lr_tree_19;
						}
						else {
							action_11;
							goto lr_tree_19;
						}
					}
				}
				else {
					if (condition_p) {
						if (condition_k) {
							if (condition_d) {
								if (condition_i) {
									action_6;
									goto lr_tree_3;
								}
								else {
									if (condition_c) {
										action_6;
										goto lr_tree_3;
									}
									else {
										action_12;
										goto lr_tree_3;
									}
								}
							}
							else {
								action_12;
								goto lr_tree_3;
							}
						}
						else {
							action_6;
							goto lr_tree_24;
						}
					}
					else {
						action_6;
						goto lr_tree_7;
					}
				}
			}
			else {
				if (condition_j) {
					if (condition_i) {
						action_4;
						goto lr_tree_19;
					}
					else {
						if (condition_c) {
							action_4;
							goto lr_tree_19;
						}
						else {
							action_7;
							goto lr_tree_19;
						}
					}
				}
				else {
					if (condition_p) {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto lr_tree_3;
								}
								else {
									action_10;
									goto lr_tree_3;
								}
							}
							else {
								if (condition_d) {
									if (condition_c) {
										action_5;
										goto lr_tree_3;
									}
									else {
										action_8;
										goto lr_tree_3;
									}
								}
								else {
									action_8;
									goto lr_tree_3;
								}
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto lr_tree_2;
							}
							else {
								action_3;
								goto lr_tree_1;
							}
						}
					}
					else {
						if (condition_i) {
							action_4;
							goto lr_tree_7;
						}
						else {
							action_3;
							goto lr_tree_7;
						}
					}
				}
			}
		}
		else {
			if (condition_p) {
				if (condition_j) {
					action_4;
					goto lr_tree_6;
				}
				else {
					if (condition_k) {
						if (condition_i) {
							if (condition_d) {
								action_5;
								goto lr_tree_3;
							}
							else {
								action_10;
								goto lr_tree_3;
							}
						}
						else {
							action_5;
							goto lr_tree_3;
						}
					}
					else {
						if (condition_i) {
							action_4;
							goto lr_tree_2;
						}
						else {
							action_2;
							goto lr_tree_1;
						}
					}
				}
			}
			else {
				action_1;
				goto lr_tree_0;
			}
		}
	lr_tree_24: lr_finish_condition(24);
		if (condition_o) {
			if (condition_j) {
				if (condition_c) {
					if (condition_g) {
						if (condition_b) {
							action_6;
							goto lr_tree_19;
						}
						else {
							action_11;
							goto lr_tree_19;
						}
					}
					else {
						action_11;
						goto lr_tree_19;
					}
				}
				else {
					action_11;
					goto lr_tree_19;
				}
			}
			else {
				if (condition_p) {
					if (condition_k) {
						if (condition_d) {
							if (condition_c) {
								if (condition_g) {
									if (condition_b) {
										action_6;
										goto lr_tree_3;
									}
									else {
										action_12;
										goto lr_tree_3;
									}
								}
								else {
									action_12;
									goto lr_tree_3;
								}
							}
							else {
								action_12;
								goto lr_tree_3;
							}
						}
						else {
							action_12;
							goto lr_tree_3;
						}
					}
					else {
						action_6;
						goto lr_tree_24;
					}
				}
				else {
					action_6;
					goto lr_tree_7;
				}
			}
		}
		else {
			if (condition_p) {
				if (condition_j) {
					action_4;
					goto lr_tree_6;
				}
				else {
					if (condition_k) {
						action_5;
						goto lr_tree_3;
					}
					else {
						action_2;
						goto lr_tree_1;
					}
				}
			}
			else {
				action_1;
				goto lr_tree_0;
			}
		}
	lr_break_0:
		if (condition_o) {
			if (condition_j) {
				if (condition_i) {
					action_4;
				}
				else {
					if (condition_h) {
						if (condition_c) {
							action_4;
						}
						else {
							action_7;
						}
					}
					else {
						action_4;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_i) {
						action_4;
					}
					else {
						if (condition_h) {
							action_3;
						}
						else {
							action_2;
						}
					}
				}
				else {
					if (condition_i) {
						action_4;
					}
					else {
						if (condition_h) {
							action_3;
						}
						else {
							action_2;
						}
					}
				}
			}
		}
		else {
			if (condition_p) {
				if (condition_j) {
					action_4;
				}
				else {
					if (condition_i) {
						action_4;
					}
					else {
						action_2;
					}
				}
			}
			else {
				action_1;
			}
		}
		continue;
	lr_break_1:
		if (condition_o) {
			if (condition_j) {
				action_11;
			}
			else {
				if (condition_p) {
					action_6;
				}
				else {
					action_6;
				}
			}
		}
		else {
			if (condition_p) {
				if (condition_j) {
					action_4;
				}
				else {
					action_2;
				}
			}
			else {
				action_1;
			}
		}
		continue;
	lr_break_2:
		if (condition_o) {
			if (condition_j) {
				if (condition_c) {
					if (condition_b) {
						action_6;
					}
					else {
						action_11;
					}
				}
				else {
					action_11;
				}
			}
			else {
				if (condition_p) {
					action_6;
				}
				else {
					action_6;
				}
			}
		}
		else {
			if (condition_p) {
				if (condition_j) {
					action_4;
				}
				else {
					action_2;
				}
			}
			else {
				action_1;
			}
		}
		continue;
	lr_break_3:
		if (condition_o) {
			if (condition_j) {
				action_6;
			}
			else {
				if (condition_p) {
					action_6;
				}
				else {
					action_6;
				}
			}
		}
		else {
			if (condition_p) {
				if (condition_j) {
					action_4;
				}
				else {
					action_4;
				}
			}
			else {
				action_1;
			}
		}
		continue;
	lr_break_6:
		if (condition_o) {
			if (condition_j) {
				if (condition_i) {
					action_6;
				}
				else {
					if (condition_c) {
						action_6;
					}
					else {
						action_11;
					}
				}
			}
			else {
				if (condition_p) {
					action_6;
				}
				else {
					action_6;
				}
			}
		}
		else {
			if (condition_p) {
				if (condition_j) {
					action_4;
				}
				else {
					if (condition_i) {
						action_4;
					}
					else {
						action_2;
					}
				}
			}
			else {
				action_1;
			}
		}
		continue;
	lr_break_7:
		if (condition_o) {
			if (condition_j) {
				action_4;
			}
			else {
				if (condition_p) {
					if (condition_i) {
						action_4;
					}
					else {
						action_2;
					}
				}
				else {
					if (condition_i) {
						action_4;
					}
					else {
						action_2;
					}
				}
			}
		}
		else {
			if (condition_p) {
				if (condition_j) {
					action_4;
				}
				else {
					if (condition_i) {
						action_4;
					}
					else {
						action_2;
					}
				}
			}
			else {
				action_1;
			}
		}
		continue;
	lr_break_19:
		if (condition_o) {
			if (condition_n) {
				if (condition_j) {
					if (condition_i) {
						action_6;
					}
					else {
						if (condition_c) {
							action_6;
						}
						else {
							action_11;
						}
					}
				}
				else {
					if (condition_p) {
						action_6;
					}
					else {
						action_6;
					}
				}
			}
			else {
				if (condition_j) {
					if (condition_i) {
						action_4;
					}
					else {
						if (condition_c) {
							action_4;
						}
						else {
							action_7;
						}
					}
				}
				else {
					if (condition_p) {
						if (condition_i) {
							action_4;
						}
						else {
							action_3;
						}
					}
					else {
						if (condition_i) {
							action_4;
						}
						else {
							action_3;
						}
					}
				}
			}
		}
		else {
			if (condition_p) {
				if (condition_j) {
					action_4;
				}
				else {
					if (condition_i) {
						action_4;
					}
					else {
						action_2;
					}
				}
			}
			else {
				action_1;
			}
		}
		continue;
	lr_break_24:
		if (condition_o) {
			if (condition_j) {
				if (condition_c) {
					if (condition_g) {
						if (condition_b) {
							action_6;
						}
						else {
							action_11;
						}
					}
					else {
						action_11;
					}
				}
				else {
					action_11;
				}
			}
			else {
				if (condition_p) {
					action_6;
				}
				else {
					action_6;
				}
			}
		}
		else {
			if (condition_p) {
				if (condition_j) {
					action_4;
				}
				else {
					action_2;
				}
			}
			else {
				action_1;
			}
		}
		continue;
	}

#undef fr_finish_condition
#undef finish_condition
#undef lr_finish_condition
}

inline static
void firstScanSpaghetti_eo(const Mat1b &img, Mat1i& imgLabels, uint* P, uint &lunique) {
	int w(img.cols), h(img.rows);
	int pix = 0, old_pix;

#define fr_finish_condition(n)	\
	c += 2; \
	old_pix = pix; \
	pix  = (img_row[c] > 0) << 31; \
	pix |= (img_row_fol[c] > 0) << 29; \
	if (c == w - 1) \
		goto fr_break_##n; \
	pix |= (img_row[c + 1] > 0) << 30; \
	pix |= (img_row_fol[c + 1] > 0) << 28; 

#define finish_condition(n) \
	c += 2; \
	old_pix = pix; \
	pix = (img_row[c] > 0) << 31; \
	pix |= (img_row_fol[c] > 0) << 29; \
	if (c == w - 1) \
		goto break_##n; \
	pix |= (img_row[c + 1] > 0) << 30; \
	pix |= (img_row_fol[c + 1] > 0) << 28; 

	for (int r = 0; r < 2; r += 2) {
		// Get row pointers
		const uchar* const img_row = img.ptr<uchar>(r);
		const uchar* const img_row_fol = (uchar *)(((char *)img_row) + img.step.p[0]);
		uint* const imgLabels_row = imgLabels.ptr<uint>(r);

		int c = -2;
		fr_finish_condition(0);
		if (condition_o) {
			action_2;
			goto fr_tree_9;
		}
		else {
			if (condition_s) {
				if (condition_p) {
					action_2;
					goto fr_tree_1;
				}
				else {
					action_2;
					goto fr_tree_3;
				}
			}
			else {
				if (condition_p) {
					action_2;
					goto fr_tree_1;
				}
				else {
					if (condition_t) {
						action_2;
						goto fr_tree_1;
					}
					else {
						action_1;
						goto fr_tree_0;
					}
				}
			}
		}
	fr_tree_0: fr_finish_condition(0);
		if (condition_o) {
			action_2;
			goto fr_tree_9;
		}
		else {
			if (condition_s) {
				if (condition_p) {
					action_2;
					goto fr_tree_1;
				}
				else {
					action_2;
					goto fr_tree_3;
				}
			}
			else {
				if (condition_p) {
					action_2;
					goto fr_tree_1;
				}
				else {
					if (condition_t) {
						action_2;
						goto fr_tree_1;
					}
					else {
						action_1;
						goto fr_tree_0;
					}
				}
			}
		}
	fr_tree_1: fr_finish_condition(1);
		if (condition_o) {
			action_6;
			goto fr_tree_9;
		}
		else {
			if (condition_s) {
				if (condition_p) {
					action_6;
					goto fr_tree_1;
				}
				else {
					action_6;
					goto fr_tree_3;
				}
			}
			else {
				if (condition_p) {
					action_2;
					goto fr_tree_1;
				}
				else {
					if (condition_t) {
						action_2;
						goto fr_tree_1;
					}
					else {
						action_1;
						goto fr_tree_0;
					}
				}
			}
		}
	fr_tree_3: fr_finish_condition(3);
		if (condition_o) {
			if (condition_r) {
				action_6;
				goto fr_tree_9;
			}
			else {
				action_2;
				goto fr_tree_9;
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_r) {
						action_6;
						goto fr_tree_1;
					}
					else {
						action_2;
						goto fr_tree_1;
					}
				}
				else {
					if (condition_r) {
						action_6;
						goto fr_tree_3;
					}
					else {
						action_2;
						goto fr_tree_3;
					}
				}
			}
			else {
				if (condition_p) {
					action_2;
					goto fr_tree_1;
				}
				else {
					if (condition_t) {
						action_2;
						goto fr_tree_1;
					}
					else {
						action_1;
						goto fr_tree_0;
					}
				}
			}
		}
	fr_tree_9: fr_finish_condition(9);
		if (condition_o) {
			if (condition_n) {
				action_6;
				goto fr_tree_9;
			}
			else {
				if (condition_r) {
					action_6;
					goto fr_tree_9;
				}
				else {
					action_2;
					goto fr_tree_9;
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_n) {
						action_6;
						goto fr_tree_1;
					}
					else {
						if (condition_r) {
							action_6;
							goto fr_tree_1;
						}
						else {
							action_2;
							goto fr_tree_1;
						}
					}
				}
				else {
					if (condition_r) {
						action_6;
						goto fr_tree_3;
					}
					else {
						if (condition_n) {
							action_6;
							goto fr_tree_3;
						}
						else {
							action_2;
							goto fr_tree_3;
						}
					}
				}
			}
			else {
				if (condition_p) {
					action_2;
					goto fr_tree_1;
				}
				else {
					if (condition_t) {
						action_2;
						goto fr_tree_1;
					}
					else {
						action_1;
						goto fr_tree_0;
					}
				}
			}
		}
	fr_break_0:
		if (condition_o) {
			action_2;
		}
		else {
			if (condition_s) {
				action_2;
			}
			else {
				action_1;
			}
		}
		continue;
	fr_break_1:
		if (condition_o) {
			action_6;
		}
		else {
			if (condition_s) {
				action_6;
			}
			else {
				action_1;
			}
		}
		continue;
	fr_break_3:
		if (condition_o) {
			if (condition_r) {
				action_6;
			}
			else {
				action_2;
			}
		}
		else {
			if (condition_s) {
				if (condition_r) {
					action_6;
				}
				else {
					action_2;
				}
			}
			else {
				action_1;
			}
		}
		continue;
	fr_break_9:
		if (condition_o) {
			if (condition_n) {
				action_6;
			}
			else {
				if (condition_r) {
					action_6;
				}
				else {
					action_2;
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_r) {
					action_6;
				}
				else {
					if (condition_n) {
						action_6;
					}
					else {
						action_2;
					}
				}
			}
			else {
				action_1;
			}
		}
		continue;
	}

	for (int r = 2; r < h - 1; r += 2) {
		// Get row pointers
		const uchar* const img_row = img.ptr<uchar>(r);
		const uchar* const img_row_fol = (uchar *)(((char *)img_row) + img.step.p[0]);
		uint* const imgLabels_row = imgLabels.ptr<uint>(r);
		uint* const imgLabels_row_prev_prev = (uint *)(((char *)imgLabels_row) - imgLabels.step.p[0] - imgLabels.step.p[0]);

		int c = -2;
		finish_condition(0);
		if (condition_o) {
			if (condition_j) {
				if (condition_i) {
					action_4;
					goto tree_73;
				}
				else {
					action_4;
					goto tree_73;
				}
			}
			else {
				if (condition_p) {
					if (condition_k) {
						if (condition_i) {
							if (condition_d) {
								action_5;
								goto tree_4;
							}
							else {
								action_10;
								goto tree_4;
							}
						}
						else {
							action_5;
							goto tree_4;
						}
					}
					else {
						if (condition_i) {
							action_4;
							goto tree_3;
						}
						else {
							action_2;
							goto tree_2;
						}
					}
				}
				else {
					if (condition_i) {
						action_4;
						goto tree_63;
					}
					else {
						action_2;
						goto tree_61;
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					action_2;
					goto tree_8;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_0: finish_condition(0);
		if (condition_o) {
			if (condition_j) {
				if (condition_i) {
					action_4;
					goto tree_73;
				}
				else {
					if (condition_h) {
						if (condition_c) {
							action_4;
							goto tree_73;
						}
						else {
							action_7;
							goto tree_73;
						}
					}
					else {
						action_4;
						goto tree_73;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_k) {
						if (condition_i) {
							if (condition_d) {
								action_5;
								goto tree_4;
							}
							else {
								action_10;
								goto tree_4;
							}
						}
						else {
							if (condition_h) {
								if (condition_d) {
									if (condition_c) {
										action_5;
										goto tree_4;
									}
									else {
										action_8;
										goto tree_4;
									}
								}
								else {
									action_8;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
					}
					else {
						if (condition_i) {
							action_4;
							goto tree_3;
						}
						else {
							if (condition_h) {
								action_3;
								goto tree_2;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					if (condition_i) {
						action_4;
						goto tree_63;
					}
					else {
						if (condition_h) {
							action_3;
							goto tree_61;
						}
						else {
							action_2;
							goto tree_61;
						}
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					action_2;
					goto tree_8;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_1: finish_condition(1);
		if (condition_o) {
			if (condition_j) {
				if (condition_i) {
					action_11;
					goto tree_73;
				}
				else {
					if (condition_h) {
						if (condition_c) {
							action_11;
							goto tree_73;
						}
						else {
							action_14;
							goto tree_73;
						}
					}
					else {
						action_11;
						goto tree_73;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_k) {
						if (condition_i) {
							if (condition_d) {
								action_12;
								goto tree_4;
							}
							else {
								action_16;
								goto tree_4;
							}
						}
						else {
							if (condition_h) {
								if (condition_d) {
									if (condition_c) {
										action_12;
										goto tree_4;
									}
									else {
										action_15;
										goto tree_4;
									}
								}
								else {
									action_15;
									goto tree_4;
								}
							}
							else {
								action_12;
								goto tree_4;
							}
						}
					}
					else {
						if (condition_h) {
							action_9;
							goto tree_47;
						}
						else {
							if (condition_i) {
								action_11;
								goto tree_3;
							}
							else {
								action_6;
								goto tree_2;
							}
						}
					}
				}
				else {
					if (condition_h) {
						action_9;
						goto tree_82;
					}
					else {
						if (condition_i) {
							action_11;
							goto tree_63;
						}
						else {
							action_6;
							goto tree_61;
						}
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						action_11;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_d) {
								action_12;
								goto tree_4;
							}
							else {
								if (condition_i) {
									action_16;
									goto tree_4;
								}
								else {
									action_12;
									goto tree_4;
								}
							}
						}
						else {
							if (condition_i) {
								action_11;
								goto tree_3;
							}
							else {
								action_6;
								goto tree_2;
							}
						}
					}
				}
				else {
					action_6;
					goto tree_8;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_2: finish_condition(2);
		if (condition_o) {
			if (condition_j) {
				action_11;
				goto tree_73;
			}
			else {
				if (condition_p) {
					if (condition_k) {
						if (condition_d) {
							action_12;
							goto tree_4;
						}
						else {
							action_12;
							goto tree_4;
						}
					}
					else {
						action_6;
						goto tree_47;
					}
				}
				else {
					action_6;
					goto tree_82;
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						action_11;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_d) {
								action_12;
								goto tree_4;
							}
							else {
								action_12;
								goto tree_4;
							}
						}
						else {
							action_6;
							goto tree_47;
						}
					}
				}
				else {
					action_6;
					goto tree_8;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							action_5;
							goto tree_4;
						}
						else {
							action_2;
							goto tree_2;
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_3: finish_condition(3);
		if (condition_o) {
			if (condition_j) {
				if (condition_c) {
					if (condition_b) {
						action_6;
						goto tree_73;
					}
					else {
						action_11;
						goto tree_73;
					}
				}
				else {
					action_11;
					goto tree_73;
				}
			}
			else {
				if (condition_p) {
					if (condition_k) {
						if (condition_d) {
							if (condition_c) {
								if (condition_b) {
									action_6;
									goto tree_4;
								}
								else {
									action_12;
									goto tree_4;
								}
							}
							else {
								action_12;
								goto tree_4;
							}
						}
						else {
							action_12;
							goto tree_4;
						}
					}
					else {
						action_6;
						goto tree_47;
					}
				}
				else {
					action_6;
					goto tree_82;
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						if (condition_c) {
							if (condition_b) {
								action_6;
								goto tree_7;
							}
							else {
								action_11;
								goto tree_7;
							}
						}
						else {
							action_11;
							goto tree_7;
						}
					}
					else {
						if (condition_k) {
							if (condition_d) {
								if (condition_c) {
									if (condition_b) {
										action_6;
										goto tree_4;
									}
									else {
										action_12;
										goto tree_4;
									}
								}
								else {
									action_12;
									goto tree_4;
								}
							}
							else {
								action_12;
								goto tree_4;
							}
						}
						else {
							action_6;
							goto tree_47;
						}
					}
				}
				else {
					action_6;
					goto tree_8;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							action_5;
							goto tree_4;
						}
						else {
							action_2;
							goto tree_2;
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_4: finish_condition(4);
		if (condition_o) {
			if (condition_j) {
				action_6;
				goto tree_73;
			}
			else {
				if (condition_p) {
					if (condition_k) {
						if (condition_d) {
							action_6;
							goto tree_4;
						}
						else {
							action_12;
							goto tree_4;
						}
					}
					else {
						action_6;
						goto tree_47;
					}
				}
				else {
					action_6;
					goto tree_82;
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						action_6;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_d) {
								action_6;
								goto tree_4;
							}
							else {
								action_12;
								goto tree_4;
							}
						}
						else {
							action_6;
							goto tree_47;
						}
					}
				}
				else {
					action_6;
					goto tree_8;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_d) {
								action_5;
								goto tree_4;
							}
							else {
								action_10;
								goto tree_4;
							}
						}
						else {
							action_4;
							goto tree_3;
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_7: finish_condition(7);
		if (condition_o) {
			if (condition_j) {
				if (condition_i) {
					action_6;
					goto tree_73;
				}
				else {
					if (condition_c) {
						action_6;
						goto tree_73;
					}
					else {
						action_11;
						goto tree_73;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_k) {
						if (condition_d) {
							if (condition_i) {
								action_6;
								goto tree_4;
							}
							else {
								if (condition_c) {
									action_6;
									goto tree_4;
								}
								else {
									action_12;
									goto tree_4;
								}
							}
						}
						else {
							action_12;
							goto tree_4;
						}
					}
					else {
						action_6;
						goto tree_47;
					}
				}
				else {
					action_6;
					goto tree_82;
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						if (condition_i) {
							action_6;
							goto tree_7;
						}
						else {
							if (condition_c) {
								action_6;
								goto tree_7;
							}
							else {
								action_11;
								goto tree_7;
							}
						}
					}
					else {
						if (condition_k) {
							if (condition_d) {
								if (condition_i) {
									action_6;
									goto tree_4;
								}
								else {
									if (condition_c) {
										action_6;
										goto tree_4;
									}
									else {
										action_12;
										goto tree_4;
									}
								}
							}
							else {
								action_12;
								goto tree_4;
							}
						}
						else {
							action_6;
							goto tree_47;
						}
					}
				}
				else {
					action_6;
					goto tree_8;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_8: finish_condition(8);
		if (condition_o) {
			if (condition_r) {
				if (condition_j) {
					if (condition_i) {
						action_11;
						goto tree_73;
					}
					else {
						if (condition_h) {
							if (condition_c) {
								action_11;
								goto tree_73;
							}
							else {
								action_14;
								goto tree_73;
							}
						}
						else {
							action_11;
							goto tree_73;
						}
					}
				}
				else {
					if (condition_p) {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_12;
									goto tree_4;
								}
								else {
									action_16;
									goto tree_4;
								}
							}
							else {
								if (condition_h) {
									if (condition_d) {
										if (condition_c) {
											action_12;
											goto tree_4;
										}
										else {
											action_15;
											goto tree_4;
										}
									}
									else {
										action_15;
										goto tree_4;
									}
								}
								else {
									action_12;
									goto tree_4;
								}
							}
						}
						else {
							if (condition_h) {
								action_9;
								goto tree_47;
							}
							else {
								if (condition_i) {
									action_11;
									goto tree_3;
								}
								else {
									action_6;
									goto tree_2;
								}
							}
						}
					}
					else {
						if (condition_h) {
							action_9;
							goto tree_82;
						}
						else {
							if (condition_i) {
								action_11;
								goto tree_63;
							}
							else {
								action_6;
								goto tree_61;
							}
						}
					}
				}
			}
			else {
				if (condition_j) {
					if (condition_i) {
						action_4;
						goto tree_73;
					}
					else {
						if (condition_h) {
							if (condition_c) {
								action_4;
								goto tree_73;
							}
							else {
								action_7;
								goto tree_73;
							}
						}
						else {
							action_4;
							goto tree_73;
						}
					}
				}
				else {
					if (condition_p) {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								if (condition_h) {
									if (condition_d) {
										if (condition_c) {
											action_5;
											goto tree_4;
										}
										else {
											action_8;
											goto tree_4;
										}
									}
									else {
										action_8;
										goto tree_4;
									}
								}
								else {
									action_5;
									goto tree_4;
								}
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								if (condition_h) {
									action_3;
									goto tree_2;
								}
								else {
									action_2;
									goto tree_2;
								}
							}
						}
					}
					else {
						if (condition_i) {
							action_4;
							goto tree_63;
						}
						else {
							if (condition_h) {
								action_3;
								goto tree_61;
							}
							else {
								action_2;
								goto tree_61;
							}
						}
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_r) {
						if (condition_j) {
							action_11;
							goto tree_7;
						}
						else {
							if (condition_k) {
								if (condition_d) {
									action_12;
									goto tree_4;
								}
								else {
									if (condition_i) {
										action_16;
										goto tree_4;
									}
									else {
										action_12;
										goto tree_4;
									}
								}
							}
							else {
								if (condition_i) {
									action_11;
									goto tree_3;
								}
								else {
									action_6;
									goto tree_2;
								}
							}
						}
					}
					else {
						if (condition_j) {
							action_4;
							goto tree_7;
						}
						else {
							if (condition_k) {
								if (condition_i) {
									if (condition_d) {
										action_5;
										goto tree_4;
									}
									else {
										action_10;
										goto tree_4;
									}
								}
								else {
									action_5;
									goto tree_4;
								}
							}
							else {
								if (condition_i) {
									action_4;
									goto tree_3;
								}
								else {
									action_2;
									goto tree_2;
								}
							}
						}
					}
				}
				else {
					if (condition_r) {
						action_6;
						goto tree_8;
					}
					else {
						action_2;
						goto tree_8;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_47: finish_condition(47);
		if (condition_o) {
			if (condition_j) {
				if (condition_c) {
					if (condition_g) {
						if (condition_b) {
							action_6;
							goto tree_73;
						}
						else {
							action_11;
							goto tree_73;
						}
					}
					else {
						action_11;
						goto tree_73;
					}
				}
				else {
					action_11;
					goto tree_73;
				}
			}
			else {
				if (condition_p) {
					if (condition_k) {
						if (condition_d) {
							if (condition_c) {
								if (condition_g) {
									if (condition_b) {
										action_6;
										goto tree_4;
									}
									else {
										action_12;
										goto tree_4;
									}
								}
								else {
									action_12;
									goto tree_4;
								}
							}
							else {
								action_12;
								goto tree_4;
							}
						}
						else {
							action_12;
							goto tree_4;
						}
					}
					else {
						action_6;
						goto tree_47;
					}
				}
				else {
					action_6;
					goto tree_82;
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						if (condition_c) {
							if (condition_g) {
								if (condition_b) {
									action_6;
									goto tree_7;
								}
								else {
									action_11;
									goto tree_7;
								}
							}
							else {
								action_11;
								goto tree_7;
							}
						}
						else {
							action_11;
							goto tree_7;
						}
					}
					else {
						if (condition_k) {
							if (condition_d) {
								if (condition_c) {
									if (condition_g) {
										if (condition_b) {
											action_6;
											goto tree_4;
										}
										else {
											action_12;
											goto tree_4;
										}
									}
									else {
										action_12;
										goto tree_4;
									}
								}
								else {
									action_12;
									goto tree_4;
								}
							}
							else {
								action_12;
								goto tree_4;
							}
						}
						else {
							action_6;
							goto tree_47;
						}
					}
				}
				else {
					action_6;
					goto tree_8;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							action_5;
							goto tree_4;
						}
						else {
							action_2;
							goto tree_2;
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_61: finish_condition(61);
		if (condition_o) {
			if (condition_r) {
				if (condition_j) {
					action_11;
					goto tree_73;
				}
				else {
					if (condition_p) {
						if (condition_k) {
							if (condition_d) {
								action_12;
								goto tree_4;
							}
							else {
								if (condition_i) {
									action_16;
									goto tree_4;
								}
								else {
									action_12;
									goto tree_4;
								}
							}
						}
						else {
							if (condition_i) {
								action_11;
								goto tree_3;
							}
							else {
								action_6;
								goto tree_2;
							}
						}
					}
					else {
						if (condition_i) {
							action_11;
							goto tree_63;
						}
						else {
							action_6;
							goto tree_61;
						}
					}
				}
			}
			else {
				if (condition_j) {
					action_4;
					goto tree_73;
				}
				else {
					if (condition_p) {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
					else {
						if (condition_i) {
							action_4;
							goto tree_63;
						}
						else {
							action_2;
							goto tree_61;
						}
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_r) {
						if (condition_j) {
							action_11;
							goto tree_7;
						}
						else {
							if (condition_k) {
								if (condition_d) {
									action_12;
									goto tree_4;
								}
								else {
									if (condition_i) {
										action_16;
										goto tree_4;
									}
									else {
										action_12;
										goto tree_4;
									}
								}
							}
							else {
								if (condition_i) {
									action_11;
									goto tree_3;
								}
								else {
									action_6;
									goto tree_2;
								}
							}
						}
					}
					else {
						if (condition_j) {
							action_4;
							goto tree_7;
						}
						else {
							if (condition_k) {
								if (condition_i) {
									if (condition_d) {
										action_5;
										goto tree_4;
									}
									else {
										action_10;
										goto tree_4;
									}
								}
								else {
									action_5;
									goto tree_4;
								}
							}
							else {
								if (condition_i) {
									action_4;
									goto tree_3;
								}
								else {
									action_2;
									goto tree_2;
								}
							}
						}
					}
				}
				else {
					if (condition_r) {
						action_6;
						goto tree_8;
					}
					else {
						action_2;
						goto tree_8;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_63: finish_condition(63);
		if (condition_o) {
			if (condition_r) {
				if (condition_j) {
					if (condition_b) {
						if (condition_i) {
							action_6;
							goto tree_73;
						}
						else {
							if (condition_c) {
								action_6;
								goto tree_73;
							}
							else {
								action_11;
								goto tree_73;
							}
						}
					}
					else {
						action_11;
						goto tree_73;
					}
				}
				else {
					if (condition_p) {
						if (condition_k) {
							if (condition_d) {
								if (condition_b) {
									if (condition_i) {
										action_6;
										goto tree_4;
									}
									else {
										if (condition_c) {
											action_6;
											goto tree_4;
										}
										else {
											action_12;
											goto tree_4;
										}
									}
								}
								else {
									action_12;
									goto tree_4;
								}
							}
							else {
								if (condition_i) {
									if (condition_b) {
										action_12;
										goto tree_4;
									}
									else {
										action_16;
										goto tree_4;
									}
								}
								else {
									action_12;
									goto tree_4;
								}
							}
						}
						else {
							if (condition_i) {
								if (condition_b) {
									action_6;
									goto tree_3;
								}
								else {
									action_11;
									goto tree_3;
								}
							}
							else {
								action_6;
								goto tree_2;
							}
						}
					}
					else {
						if (condition_i) {
							if (condition_b) {
								action_6;
								goto tree_63;
							}
							else {
								action_11;
								goto tree_63;
							}
						}
						else {
							action_6;
							goto tree_61;
						}
					}
				}
			}
			else {
				if (condition_j) {
					action_4;
					goto tree_73;
				}
				else {
					if (condition_p) {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
					else {
						if (condition_i) {
							action_4;
							goto tree_63;
						}
						else {
							action_2;
							goto tree_61;
						}
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_r) {
						if (condition_j) {
							if (condition_b) {
								if (condition_i) {
									action_6;
									goto tree_7;
								}
								else {
									if (condition_c) {
										action_6;
										goto tree_7;
									}
									else {
										action_11;
										goto tree_7;
									}
								}
							}
							else {
								action_11;
								goto tree_7;
							}
						}
						else {
							if (condition_k) {
								if (condition_d) {
									if (condition_b) {
										if (condition_i) {
											action_6;
											goto tree_4;
										}
										else {
											if (condition_c) {
												action_6;
												goto tree_4;
											}
											else {
												action_12;
												goto tree_4;
											}
										}
									}
									else {
										action_12;
										goto tree_4;
									}
								}
								else {
									if (condition_i) {
										if (condition_b) {
											action_12;
											goto tree_4;
										}
										else {
											action_16;
											goto tree_4;
										}
									}
									else {
										action_12;
										goto tree_4;
									}
								}
							}
							else {
								if (condition_i) {
									if (condition_b) {
										action_6;
										goto tree_3;
									}
									else {
										action_11;
										goto tree_3;
									}
								}
								else {
									action_6;
									goto tree_2;
								}
							}
						}
					}
					else {
						if (condition_j) {
							action_4;
							goto tree_7;
						}
						else {
							if (condition_k) {
								if (condition_i) {
									if (condition_d) {
										action_5;
										goto tree_4;
									}
									else {
										action_10;
										goto tree_4;
									}
								}
								else {
									action_5;
									goto tree_4;
								}
							}
							else {
								if (condition_i) {
									action_4;
									goto tree_3;
								}
								else {
									action_2;
									goto tree_2;
								}
							}
						}
					}
				}
				else {
					if (condition_r) {
						action_6;
						goto tree_8;
					}
					else {
						action_2;
						goto tree_8;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_73: finish_condition(73);
		if (condition_o) {
			if (condition_n) {
				if (condition_j) {
					if (condition_i) {
						action_6;
						goto tree_73;
					}
					else {
						if (condition_c) {
							action_6;
							goto tree_73;
						}
						else {
							action_11;
							goto tree_73;
						}
					}
				}
				else {
					if (condition_p) {
						if (condition_k) {
							if (condition_d) {
								if (condition_i) {
									action_6;
									goto tree_4;
								}
								else {
									if (condition_c) {
										action_6;
										goto tree_4;
									}
									else {
										action_12;
										goto tree_4;
									}
								}
							}
							else {
								action_12;
								goto tree_4;
							}
						}
						else {
							action_6;
							goto tree_47;
						}
					}
					else {
						action_6;
						goto tree_82;
					}
				}
			}
			else {
				if (condition_r) {
					if (condition_j) {
						if (condition_i) {
							action_6;
							goto tree_73;
						}
						else {
							if (condition_c) {
								action_6;
								goto tree_73;
							}
							else {
								action_11;
								goto tree_73;
							}
						}
					}
					else {
						if (condition_p) {
							if (condition_k) {
								if (condition_d) {
									if (condition_i) {
										action_6;
										goto tree_4;
									}
									else {
										if (condition_c) {
											action_6;
											goto tree_4;
										}
										else {
											action_12;
											goto tree_4;
										}
									}
								}
								else {
									action_12;
									goto tree_4;
								}
							}
							else {
								action_6;
								goto tree_47;
							}
						}
						else {
							action_6;
							goto tree_82;
						}
					}
				}
				else {
					if (condition_j) {
						if (condition_i) {
							action_4;
							goto tree_73;
						}
						else {
							if (condition_c) {
								action_4;
								goto tree_73;
							}
							else {
								action_7;
								goto tree_73;
							}
						}
					}
					else {
						if (condition_p) {
							if (condition_k) {
								if (condition_i) {
									if (condition_d) {
										action_5;
										goto tree_4;
									}
									else {
										action_10;
										goto tree_4;
									}
								}
								else {
									if (condition_d) {
										if (condition_c) {
											action_5;
											goto tree_4;
										}
										else {
											action_8;
											goto tree_4;
										}
									}
									else {
										action_8;
										goto tree_4;
									}
								}
							}
							else {
								if (condition_i) {
									action_4;
									goto tree_3;
								}
								else {
									action_3;
									goto tree_2;
								}
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_63;
							}
							else {
								action_3;
								goto tree_61;
							}
						}
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_n) {
						if (condition_j) {
							if (condition_i) {
								action_6;
								goto tree_7;
							}
							else {
								if (condition_c) {
									action_6;
									goto tree_7;
								}
								else {
									action_11;
									goto tree_7;
								}
							}
						}
						else {
							if (condition_k) {
								if (condition_d) {
									if (condition_i) {
										action_6;
										goto tree_4;
									}
									else {
										if (condition_c) {
											action_6;
											goto tree_4;
										}
										else {
											action_12;
											goto tree_4;
										}
									}
								}
								else {
									action_12;
									goto tree_4;
								}
							}
							else {
								action_6;
								goto tree_47;
							}
						}
					}
					else {
						if (condition_r) {
							if (condition_j) {
								if (condition_i) {
									action_6;
									goto tree_7;
								}
								else {
									if (condition_c) {
										action_6;
										goto tree_7;
									}
									else {
										action_11;
										goto tree_7;
									}
								}
							}
							else {
								if (condition_k) {
									if (condition_d) {
										if (condition_i) {
											action_6;
											goto tree_4;
										}
										else {
											if (condition_c) {
												action_6;
												goto tree_4;
											}
											else {
												action_12;
												goto tree_4;
											}
										}
									}
									else {
										action_12;
										goto tree_4;
									}
								}
								else {
									if (condition_i) {
										action_6;
										goto tree_3;
									}
									else {
										action_6;
										goto tree_2;
									}
								}
							}
						}
						else {
							if (condition_j) {
								action_4;
								goto tree_7;
							}
							else {
								if (condition_k) {
									if (condition_i) {
										if (condition_d) {
											action_5;
											goto tree_4;
										}
										else {
											action_10;
											goto tree_4;
										}
									}
									else {
										action_5;
										goto tree_4;
									}
								}
								else {
									if (condition_i) {
										action_4;
										goto tree_3;
									}
									else {
										action_2;
										goto tree_2;
									}
								}
							}
						}
					}
				}
				else {
					if (condition_r) {
						action_6;
						goto tree_8;
					}
					else {
						if (condition_n) {
							action_6;
							goto tree_8;
						}
						else {
							action_2;
							goto tree_8;
						}
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_82: finish_condition(82);
		if (condition_o) {
			if (condition_r) {
				if (condition_j) {
					if (condition_g) {
						if (condition_b) {
							if (condition_i) {
								action_6;
								goto tree_73;
							}
							else {
								if (condition_c) {
									action_6;
									goto tree_73;
								}
								else {
									action_11;
									goto tree_73;
								}
							}
						}
						else {
							action_11;
							goto tree_73;
						}
					}
					else {
						action_11;
						goto tree_73;
					}
				}
				else {
					if (condition_p) {
						if (condition_k) {
							if (condition_d) {
								if (condition_g) {
									if (condition_b) {
										if (condition_i) {
											action_6;
											goto tree_4;
										}
										else {
											if (condition_c) {
												action_6;
												goto tree_4;
											}
											else {
												action_12;
												goto tree_4;
											}
										}
									}
									else {
										action_12;
										goto tree_4;
									}
								}
								else {
									action_12;
									goto tree_4;
								}
							}
							else {
								if (condition_i) {
									if (condition_g) {
										if (condition_b) {
											action_12;
											goto tree_4;
										}
										else {
											action_16;
											goto tree_4;
										}
									}
									else {
										action_16;
										goto tree_4;
									}
								}
								else {
									action_12;
									goto tree_4;
								}
							}
						}
						else {
							if (condition_i) {
								if (condition_g) {
									if (condition_b) {
										action_6;
										goto tree_3;
									}
									else {
										action_11;
										goto tree_3;
									}
								}
								else {
									action_11;
									goto tree_3;
								}
							}
							else {
								action_6;
								goto tree_2;
							}
						}
					}
					else {
						if (condition_i) {
							if (condition_g) {
								if (condition_b) {
									action_6;
									goto tree_63;
								}
								else {
									action_11;
									goto tree_63;
								}
							}
							else {
								action_11;
								goto tree_63;
							}
						}
						else {
							action_6;
							goto tree_61;
						}
					}
				}
			}
			else {
				if (condition_j) {
					action_4;
					goto tree_73;
				}
				else {
					if (condition_p) {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
					else {
						if (condition_i) {
							action_4;
							goto tree_63;
						}
						else {
							action_2;
							goto tree_61;
						}
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_r) {
						if (condition_j) {
							if (condition_g) {
								if (condition_b) {
									if (condition_i) {
										action_6;
										goto tree_7;
									}
									else {
										if (condition_c) {
											action_6;
											goto tree_7;
										}
										else {
											action_11;
											goto tree_7;
										}
									}
								}
								else {
									action_11;
									goto tree_7;
								}
							}
							else {
								action_11;
								goto tree_7;
							}
						}
						else {
							if (condition_k) {
								if (condition_d) {
									if (condition_g) {
										if (condition_b) {
											if (condition_i) {
												action_6;
												goto tree_4;
											}
											else {
												if (condition_c) {
													action_6;
													goto tree_4;
												}
												else {
													action_12;
													goto tree_4;
												}
											}
										}
										else {
											action_12;
											goto tree_4;
										}
									}
									else {
										action_12;
										goto tree_4;
									}
								}
								else {
									if (condition_i) {
										if (condition_g) {
											if (condition_b) {
												action_12;
												goto tree_4;
											}
											else {
												action_16;
												goto tree_4;
											}
										}
										else {
											action_16;
											goto tree_4;
										}
									}
									else {
										action_12;
										goto tree_4;
									}
								}
							}
							else {
								if (condition_i) {
									if (condition_g) {
										if (condition_b) {
											action_6;
											goto tree_3;
										}
										else {
											action_11;
											goto tree_3;
										}
									}
									else {
										action_11;
										goto tree_3;
									}
								}
								else {
									action_6;
									goto tree_2;
								}
							}
						}
					}
					else {
						if (condition_j) {
							action_4;
							goto tree_7;
						}
						else {
							if (condition_k) {
								if (condition_i) {
									if (condition_d) {
										action_5;
										goto tree_4;
									}
									else {
										action_10;
										goto tree_4;
									}
								}
								else {
									action_5;
									goto tree_4;
								}
							}
							else {
								if (condition_i) {
									action_4;
									goto tree_3;
								}
								else {
									action_2;
									goto tree_2;
								}
							}
						}
					}
				}
				else {
					if (condition_r) {
						action_6;
						goto tree_8;
					}
					else {
						action_2;
						goto tree_8;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	break_0:
		if (condition_o) {
			if (condition_i) {
				action_4;
			}
			else {
				if (condition_h) {
					action_3;
				}
				else {
					action_2;
				}
			}
		}
		else {
			if (condition_s) {
				action_2;
			}
			else {
				action_1;
			}
		}
		continue;
	break_1:
		if (condition_o) {
			if (condition_h) {
				action_9;
			}
			else {
				if (condition_i) {
					action_11;
				}
				else {
					action_6;
				}
			}
		}
		else {
			if (condition_s) {
				action_6;
			}
			else {
				action_1;
			}
		}
		continue;
	break_2:
		if (condition_o) {
			action_6;
		}
		else {
			if (condition_s) {
				action_6;
			}
			else {
				action_1;
			}
		}
		continue;
	break_3:
		if (condition_o) {
			action_6;
		}
		else {
			if (condition_s) {
				action_6;
			}
			else {
				action_1;
			}
		}
		continue;
	break_4:
		if (condition_o) {
			action_6;
		}
		else {
			if (condition_s) {
				action_6;
			}
			else {
				action_1;
			}
		}
		continue;
	break_7:
		if (condition_o) {
			action_6;
		}
		else {
			if (condition_s) {
				action_6;
			}
			else {
				action_1;
			}
		}
		continue;
	break_8:
		if (condition_o) {
			if (condition_r) {
				if (condition_h) {
					action_9;
				}
				else {
					if (condition_i) {
						action_11;
					}
					else {
						action_6;
					}
				}
			}
			else {
				if (condition_i) {
					action_4;
				}
				else {
					if (condition_h) {
						action_3;
					}
					else {
						action_2;
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_r) {
					action_6;
				}
				else {
					action_2;
				}
			}
			else {
				action_1;
			}
		}
		continue;
	break_47:
		if (condition_o) {
			action_6;
		}
		else {
			if (condition_s) {
				action_6;
			}
			else {
				action_1;
			}
		}
		continue;
	break_61:
		if (condition_o) {
			if (condition_r) {
				if (condition_i) {
					action_11;
				}
				else {
					action_6;
				}
			}
			else {
				if (condition_i) {
					action_4;
				}
				else {
					action_2;
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_r) {
					action_6;
				}
				else {
					action_2;
				}
			}
			else {
				action_1;
			}
		}
		continue;
	break_63:
		if (condition_o) {
			if (condition_r) {
				if (condition_i) {
					if (condition_b) {
						action_6;
					}
					else {
						action_11;
					}
				}
				else {
					action_6;
				}
			}
			else {
				if (condition_i) {
					action_4;
				}
				else {
					action_2;
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_r) {
					action_6;
				}
				else {
					action_2;
				}
			}
			else {
				action_1;
			}
		}
		continue;
	break_73:
		if (condition_o) {
			if (condition_n) {
				action_6;
			}
			else {
				if (condition_r) {
					action_6;
				}
				else {
					if (condition_i) {
						action_4;
					}
					else {
						action_3;
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_r) {
					action_6;
				}
				else {
					if (condition_n) {
						action_6;
					}
					else {
						action_2;
					}
				}
			}
			else {
				action_1;
			}
		}
		continue;
	break_82:
		if (condition_o) {
			if (condition_r) {
				if (condition_i) {
					if (condition_g) {
						if (condition_b) {
							action_6;
						}
						else {
							action_11;
						}
					}
					else {
						action_11;
					}
				}
				else {
					action_6;
				}
			}
			else {
				if (condition_i) {
					action_4;
				}
				else {
					action_2;
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_r) {
					action_6;
				}
				else {
					action_2;
				}
			}
			else {
				action_1;
			}
		}
		continue;
	}

#undef fr_finish_condition
#undef finish_condition
}

inline static
void firstScanSpaghetti_oo(const Mat1b &img, Mat1i& imgLabels, uint* P, uint &lunique) {
	int w(img.cols), h(img.rows);
	int pix = 0, old_pix;

#define fr_finish_condition(n)	\
	c += 2; \
	old_pix = pix; \
	pix  = (img_row[c] > 0) << 31; \
	pix |= (img_row_fol[c] > 0) << 29; \
	if (c == w - 1) \
		goto fr_break_##n; \
	pix |= (img_row[c + 1] > 0) << 30; \
	pix |= (img_row_fol[c + 1] > 0) << 28; 

#define finish_condition(n) \
	c += 2; \
	old_pix = pix; \
	pix = (img_row[c] > 0) << 31; \
	pix |= (img_row_fol[c] > 0) << 29; \
	if (c == w - 1) \
		goto break_##n; \
	pix |= (img_row[c + 1] > 0) << 30; \
	pix |= (img_row_fol[c + 1] > 0) << 28; 

#define lr_finish_condition(n) \
	c += 2; \
	old_pix = pix; \
	pix = (img_row[c] > 0) << 31; \
	if (c == w - 1) \
		goto lr_break_##n; \
	pix |= (img_row[c + 1] > 0) << 30; \

	for (int r = 0; r < 2; r += 2) {
		// Get row pointers
		const uchar* const img_row = img.ptr<uchar>(r);
		const uchar* const img_row_fol = (uchar *)(((char *)img_row) + img.step.p[0]);
		uint* const imgLabels_row = imgLabels.ptr<uint>(r);

		int c = -2;
		fr_finish_condition(0);
		if (condition_o) {
			action_2;
			goto fr_tree_9;
		}
		else {
			if (condition_s) {
				if (condition_p) {
					action_2;
					goto fr_tree_1;
				}
				else {
					action_2;
					goto fr_tree_3;
				}
			}
			else {
				if (condition_p) {
					action_2;
					goto fr_tree_1;
				}
				else {
					if (condition_t) {
						action_2;
						goto fr_tree_1;
					}
					else {
						action_1;
						goto fr_tree_0;
					}
				}
			}
		}
	fr_tree_0: fr_finish_condition(0);
		if (condition_o) {
			action_2;
			goto fr_tree_9;
		}
		else {
			if (condition_s) {
				if (condition_p) {
					action_2;
					goto fr_tree_1;
				}
				else {
					action_2;
					goto fr_tree_3;
				}
			}
			else {
				if (condition_p) {
					action_2;
					goto fr_tree_1;
				}
				else {
					if (condition_t) {
						action_2;
						goto fr_tree_1;
					}
					else {
						action_1;
						goto fr_tree_0;
					}
				}
			}
		}
	fr_tree_1: fr_finish_condition(1);
		if (condition_o) {
			action_6;
			goto fr_tree_9;
		}
		else {
			if (condition_s) {
				if (condition_p) {
					action_6;
					goto fr_tree_1;
				}
				else {
					action_6;
					goto fr_tree_3;
				}
			}
			else {
				if (condition_p) {
					action_2;
					goto fr_tree_1;
				}
				else {
					if (condition_t) {
						action_2;
						goto fr_tree_1;
					}
					else {
						action_1;
						goto fr_tree_0;
					}
				}
			}
		}
	fr_tree_3: fr_finish_condition(3);
		if (condition_o) {
			if (condition_r) {
				action_6;
				goto fr_tree_9;
			}
			else {
				action_2;
				goto fr_tree_9;
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_r) {
						action_6;
						goto fr_tree_1;
					}
					else {
						action_2;
						goto fr_tree_1;
					}
				}
				else {
					if (condition_r) {
						action_6;
						goto fr_tree_3;
					}
					else {
						action_2;
						goto fr_tree_3;
					}
				}
			}
			else {
				if (condition_p) {
					action_2;
					goto fr_tree_1;
				}
				else {
					if (condition_t) {
						action_2;
						goto fr_tree_1;
					}
					else {
						action_1;
						goto fr_tree_0;
					}
				}
			}
		}
	fr_tree_9: fr_finish_condition(9);
		if (condition_o) {
			if (condition_n) {
				action_6;
				goto fr_tree_9;
			}
			else {
				if (condition_r) {
					action_6;
					goto fr_tree_9;
				}
				else {
					action_2;
					goto fr_tree_9;
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_n) {
						action_6;
						goto fr_tree_1;
					}
					else {
						if (condition_r) {
							action_6;
							goto fr_tree_1;
						}
						else {
							action_2;
							goto fr_tree_1;
						}
					}
				}
				else {
					if (condition_r) {
						action_6;
						goto fr_tree_3;
					}
					else {
						if (condition_n) {
							action_6;
							goto fr_tree_3;
						}
						else {
							action_2;
							goto fr_tree_3;
						}
					}
				}
			}
			else {
				if (condition_p) {
					action_2;
					goto fr_tree_1;
				}
				else {
					if (condition_t) {
						action_2;
						goto fr_tree_1;
					}
					else {
						action_1;
						goto fr_tree_0;
					}
				}
			}
		}
	fr_break_0:
		if (condition_o) {
			action_2;
		}
		else {
			if (condition_s) {
				action_2;
			}
			else {
				action_1;
			}
		}
		continue;
	fr_break_1:
		if (condition_o) {
			action_6;
		}
		else {
			if (condition_s) {
				action_6;
			}
			else {
				action_1;
			}
		}
		continue;
	fr_break_3:
		if (condition_o) {
			if (condition_r) {
				action_6;
			}
			else {
				action_2;
			}
		}
		else {
			if (condition_s) {
				if (condition_r) {
					action_6;
				}
				else {
					action_2;
				}
			}
			else {
				action_1;
			}
		}
		continue;
	fr_break_9:
		if (condition_o) {
			if (condition_n) {
				action_6;
			}
			else {
				if (condition_r) {
					action_6;
				}
				else {
					action_2;
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_r) {
					action_6;
				}
				else {
					if (condition_n) {
						action_6;
					}
					else {
						action_2;
					}
				}
			}
			else {
				action_1;
			}
		}
		continue;
	}

	for (int r = 2; r < h - 1; r += 2) {
		// Get row pointers
		const uchar* const img_row = img.ptr<uchar>(r);
		const uchar* const img_row_fol = (uchar *)(((char *)img_row) + img.step.p[0]);
		uint* const imgLabels_row = imgLabels.ptr<uint>(r);
		uint* const imgLabels_row_prev_prev = (uint *)(((char *)imgLabels_row) - imgLabels.step.p[0] - imgLabels.step.p[0]);

		int c = -2;
		finish_condition(0);
		if (condition_o) {
			if (condition_j) {
				if (condition_i) {
					action_4;
					goto tree_73;
				}
				else {
					action_4;
					goto tree_73;
				}
			}
			else {
				if (condition_p) {
					if (condition_k) {
						if (condition_i) {
							if (condition_d) {
								action_5;
								goto tree_4;
							}
							else {
								action_10;
								goto tree_4;
							}
						}
						else {
							action_5;
							goto tree_4;
						}
					}
					else {
						if (condition_i) {
							action_4;
							goto tree_3;
						}
						else {
							action_2;
							goto tree_2;
						}
					}
				}
				else {
					if (condition_i) {
						action_4;
						goto tree_63;
					}
					else {
						action_2;
						goto tree_61;
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					action_2;
					goto tree_8;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_0: finish_condition(0);
		if (condition_o) {
			if (condition_j) {
				if (condition_i) {
					action_4;
					goto tree_73;
				}
				else {
					if (condition_h) {
						if (condition_c) {
							action_4;
							goto tree_73;
						}
						else {
							action_7;
							goto tree_73;
						}
					}
					else {
						action_4;
						goto tree_73;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_k) {
						if (condition_i) {
							if (condition_d) {
								action_5;
								goto tree_4;
							}
							else {
								action_10;
								goto tree_4;
							}
						}
						else {
							if (condition_h) {
								if (condition_d) {
									if (condition_c) {
										action_5;
										goto tree_4;
									}
									else {
										action_8;
										goto tree_4;
									}
								}
								else {
									action_8;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
					}
					else {
						if (condition_i) {
							action_4;
							goto tree_3;
						}
						else {
							if (condition_h) {
								action_3;
								goto tree_2;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					if (condition_i) {
						action_4;
						goto tree_63;
					}
					else {
						if (condition_h) {
							action_3;
							goto tree_61;
						}
						else {
							action_2;
							goto tree_61;
						}
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					action_2;
					goto tree_8;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_1: finish_condition(1);
		if (condition_o) {
			if (condition_j) {
				if (condition_i) {
					action_11;
					goto tree_73;
				}
				else {
					if (condition_h) {
						if (condition_c) {
							action_11;
							goto tree_73;
						}
						else {
							action_14;
							goto tree_73;
						}
					}
					else {
						action_11;
						goto tree_73;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_k) {
						if (condition_i) {
							if (condition_d) {
								action_12;
								goto tree_4;
							}
							else {
								action_16;
								goto tree_4;
							}
						}
						else {
							if (condition_h) {
								if (condition_d) {
									if (condition_c) {
										action_12;
										goto tree_4;
									}
									else {
										action_15;
										goto tree_4;
									}
								}
								else {
									action_15;
									goto tree_4;
								}
							}
							else {
								action_12;
								goto tree_4;
							}
						}
					}
					else {
						if (condition_h) {
							action_9;
							goto tree_47;
						}
						else {
							if (condition_i) {
								action_11;
								goto tree_3;
							}
							else {
								action_6;
								goto tree_2;
							}
						}
					}
				}
				else {
					if (condition_h) {
						action_9;
						goto tree_82;
					}
					else {
						if (condition_i) {
							action_11;
							goto tree_63;
						}
						else {
							action_6;
							goto tree_61;
						}
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						action_11;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_d) {
								action_12;
								goto tree_4;
							}
							else {
								if (condition_i) {
									action_16;
									goto tree_4;
								}
								else {
									action_12;
									goto tree_4;
								}
							}
						}
						else {
							if (condition_i) {
								action_11;
								goto tree_3;
							}
							else {
								action_6;
								goto tree_2;
							}
						}
					}
				}
				else {
					action_6;
					goto tree_8;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_2: finish_condition(2);
		if (condition_o) {
			if (condition_j) {
				action_11;
				goto tree_73;
			}
			else {
				if (condition_p) {
					if (condition_k) {
						if (condition_d) {
							action_12;
							goto tree_4;
						}
						else {
							action_12;
							goto tree_4;
						}
					}
					else {
						action_6;
						goto tree_47;
					}
				}
				else {
					action_6;
					goto tree_82;
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						action_11;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_d) {
								action_12;
								goto tree_4;
							}
							else {
								action_12;
								goto tree_4;
							}
						}
						else {
							action_6;
							goto tree_47;
						}
					}
				}
				else {
					action_6;
					goto tree_8;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							action_5;
							goto tree_4;
						}
						else {
							action_2;
							goto tree_2;
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_3: finish_condition(3);
		if (condition_o) {
			if (condition_j) {
				if (condition_c) {
					if (condition_b) {
						action_6;
						goto tree_73;
					}
					else {
						action_11;
						goto tree_73;
					}
				}
				else {
					action_11;
					goto tree_73;
				}
			}
			else {
				if (condition_p) {
					if (condition_k) {
						if (condition_d) {
							if (condition_c) {
								if (condition_b) {
									action_6;
									goto tree_4;
								}
								else {
									action_12;
									goto tree_4;
								}
							}
							else {
								action_12;
								goto tree_4;
							}
						}
						else {
							action_12;
							goto tree_4;
						}
					}
					else {
						action_6;
						goto tree_47;
					}
				}
				else {
					action_6;
					goto tree_82;
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						if (condition_c) {
							if (condition_b) {
								action_6;
								goto tree_7;
							}
							else {
								action_11;
								goto tree_7;
							}
						}
						else {
							action_11;
							goto tree_7;
						}
					}
					else {
						if (condition_k) {
							if (condition_d) {
								if (condition_c) {
									if (condition_b) {
										action_6;
										goto tree_4;
									}
									else {
										action_12;
										goto tree_4;
									}
								}
								else {
									action_12;
									goto tree_4;
								}
							}
							else {
								action_12;
								goto tree_4;
							}
						}
						else {
							action_6;
							goto tree_47;
						}
					}
				}
				else {
					action_6;
					goto tree_8;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							action_5;
							goto tree_4;
						}
						else {
							action_2;
							goto tree_2;
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_4: finish_condition(4);
		if (condition_o) {
			if (condition_j) {
				action_6;
				goto tree_73;
			}
			else {
				if (condition_p) {
					if (condition_k) {
						if (condition_d) {
							action_6;
							goto tree_4;
						}
						else {
							action_12;
							goto tree_4;
						}
					}
					else {
						action_6;
						goto tree_47;
					}
				}
				else {
					action_6;
					goto tree_82;
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						action_6;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_d) {
								action_6;
								goto tree_4;
							}
							else {
								action_12;
								goto tree_4;
							}
						}
						else {
							action_6;
							goto tree_47;
						}
					}
				}
				else {
					action_6;
					goto tree_8;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_d) {
								action_5;
								goto tree_4;
							}
							else {
								action_10;
								goto tree_4;
							}
						}
						else {
							action_4;
							goto tree_3;
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_7: finish_condition(7);
		if (condition_o) {
			if (condition_j) {
				if (condition_i) {
					action_6;
					goto tree_73;
				}
				else {
					if (condition_c) {
						action_6;
						goto tree_73;
					}
					else {
						action_11;
						goto tree_73;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_k) {
						if (condition_d) {
							if (condition_i) {
								action_6;
								goto tree_4;
							}
							else {
								if (condition_c) {
									action_6;
									goto tree_4;
								}
								else {
									action_12;
									goto tree_4;
								}
							}
						}
						else {
							action_12;
							goto tree_4;
						}
					}
					else {
						action_6;
						goto tree_47;
					}
				}
				else {
					action_6;
					goto tree_82;
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						if (condition_i) {
							action_6;
							goto tree_7;
						}
						else {
							if (condition_c) {
								action_6;
								goto tree_7;
							}
							else {
								action_11;
								goto tree_7;
							}
						}
					}
					else {
						if (condition_k) {
							if (condition_d) {
								if (condition_i) {
									action_6;
									goto tree_4;
								}
								else {
									if (condition_c) {
										action_6;
										goto tree_4;
									}
									else {
										action_12;
										goto tree_4;
									}
								}
							}
							else {
								action_12;
								goto tree_4;
							}
						}
						else {
							action_6;
							goto tree_47;
						}
					}
				}
				else {
					action_6;
					goto tree_8;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_8: finish_condition(8);
		if (condition_o) {
			if (condition_r) {
				if (condition_j) {
					if (condition_i) {
						action_11;
						goto tree_73;
					}
					else {
						if (condition_h) {
							if (condition_c) {
								action_11;
								goto tree_73;
							}
							else {
								action_14;
								goto tree_73;
							}
						}
						else {
							action_11;
							goto tree_73;
						}
					}
				}
				else {
					if (condition_p) {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_12;
									goto tree_4;
								}
								else {
									action_16;
									goto tree_4;
								}
							}
							else {
								if (condition_h) {
									if (condition_d) {
										if (condition_c) {
											action_12;
											goto tree_4;
										}
										else {
											action_15;
											goto tree_4;
										}
									}
									else {
										action_15;
										goto tree_4;
									}
								}
								else {
									action_12;
									goto tree_4;
								}
							}
						}
						else {
							if (condition_h) {
								action_9;
								goto tree_47;
							}
							else {
								if (condition_i) {
									action_11;
									goto tree_3;
								}
								else {
									action_6;
									goto tree_2;
								}
							}
						}
					}
					else {
						if (condition_h) {
							action_9;
							goto tree_82;
						}
						else {
							if (condition_i) {
								action_11;
								goto tree_63;
							}
							else {
								action_6;
								goto tree_61;
							}
						}
					}
				}
			}
			else {
				if (condition_j) {
					if (condition_i) {
						action_4;
						goto tree_73;
					}
					else {
						if (condition_h) {
							if (condition_c) {
								action_4;
								goto tree_73;
							}
							else {
								action_7;
								goto tree_73;
							}
						}
						else {
							action_4;
							goto tree_73;
						}
					}
				}
				else {
					if (condition_p) {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								if (condition_h) {
									if (condition_d) {
										if (condition_c) {
											action_5;
											goto tree_4;
										}
										else {
											action_8;
											goto tree_4;
										}
									}
									else {
										action_8;
										goto tree_4;
									}
								}
								else {
									action_5;
									goto tree_4;
								}
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								if (condition_h) {
									action_3;
									goto tree_2;
								}
								else {
									action_2;
									goto tree_2;
								}
							}
						}
					}
					else {
						if (condition_i) {
							action_4;
							goto tree_63;
						}
						else {
							if (condition_h) {
								action_3;
								goto tree_61;
							}
							else {
								action_2;
								goto tree_61;
							}
						}
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_r) {
						if (condition_j) {
							action_11;
							goto tree_7;
						}
						else {
							if (condition_k) {
								if (condition_d) {
									action_12;
									goto tree_4;
								}
								else {
									if (condition_i) {
										action_16;
										goto tree_4;
									}
									else {
										action_12;
										goto tree_4;
									}
								}
							}
							else {
								if (condition_i) {
									action_11;
									goto tree_3;
								}
								else {
									action_6;
									goto tree_2;
								}
							}
						}
					}
					else {
						if (condition_j) {
							action_4;
							goto tree_7;
						}
						else {
							if (condition_k) {
								if (condition_i) {
									if (condition_d) {
										action_5;
										goto tree_4;
									}
									else {
										action_10;
										goto tree_4;
									}
								}
								else {
									action_5;
									goto tree_4;
								}
							}
							else {
								if (condition_i) {
									action_4;
									goto tree_3;
								}
								else {
									action_2;
									goto tree_2;
								}
							}
						}
					}
				}
				else {
					if (condition_r) {
						action_6;
						goto tree_8;
					}
					else {
						action_2;
						goto tree_8;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_47: finish_condition(47);
		if (condition_o) {
			if (condition_j) {
				if (condition_c) {
					if (condition_g) {
						if (condition_b) {
							action_6;
							goto tree_73;
						}
						else {
							action_11;
							goto tree_73;
						}
					}
					else {
						action_11;
						goto tree_73;
					}
				}
				else {
					action_11;
					goto tree_73;
				}
			}
			else {
				if (condition_p) {
					if (condition_k) {
						if (condition_d) {
							if (condition_c) {
								if (condition_g) {
									if (condition_b) {
										action_6;
										goto tree_4;
									}
									else {
										action_12;
										goto tree_4;
									}
								}
								else {
									action_12;
									goto tree_4;
								}
							}
							else {
								action_12;
								goto tree_4;
							}
						}
						else {
							action_12;
							goto tree_4;
						}
					}
					else {
						action_6;
						goto tree_47;
					}
				}
				else {
					action_6;
					goto tree_82;
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						if (condition_c) {
							if (condition_g) {
								if (condition_b) {
									action_6;
									goto tree_7;
								}
								else {
									action_11;
									goto tree_7;
								}
							}
							else {
								action_11;
								goto tree_7;
							}
						}
						else {
							action_11;
							goto tree_7;
						}
					}
					else {
						if (condition_k) {
							if (condition_d) {
								if (condition_c) {
									if (condition_g) {
										if (condition_b) {
											action_6;
											goto tree_4;
										}
										else {
											action_12;
											goto tree_4;
										}
									}
									else {
										action_12;
										goto tree_4;
									}
								}
								else {
									action_12;
									goto tree_4;
								}
							}
							else {
								action_12;
								goto tree_4;
							}
						}
						else {
							action_6;
							goto tree_47;
						}
					}
				}
				else {
					action_6;
					goto tree_8;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							action_5;
							goto tree_4;
						}
						else {
							action_2;
							goto tree_2;
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_61: finish_condition(61);
		if (condition_o) {
			if (condition_r) {
				if (condition_j) {
					action_11;
					goto tree_73;
				}
				else {
					if (condition_p) {
						if (condition_k) {
							if (condition_d) {
								action_12;
								goto tree_4;
							}
							else {
								if (condition_i) {
									action_16;
									goto tree_4;
								}
								else {
									action_12;
									goto tree_4;
								}
							}
						}
						else {
							if (condition_i) {
								action_11;
								goto tree_3;
							}
							else {
								action_6;
								goto tree_2;
							}
						}
					}
					else {
						if (condition_i) {
							action_11;
							goto tree_63;
						}
						else {
							action_6;
							goto tree_61;
						}
					}
				}
			}
			else {
				if (condition_j) {
					action_4;
					goto tree_73;
				}
				else {
					if (condition_p) {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
					else {
						if (condition_i) {
							action_4;
							goto tree_63;
						}
						else {
							action_2;
							goto tree_61;
						}
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_r) {
						if (condition_j) {
							action_11;
							goto tree_7;
						}
						else {
							if (condition_k) {
								if (condition_d) {
									action_12;
									goto tree_4;
								}
								else {
									if (condition_i) {
										action_16;
										goto tree_4;
									}
									else {
										action_12;
										goto tree_4;
									}
								}
							}
							else {
								if (condition_i) {
									action_11;
									goto tree_3;
								}
								else {
									action_6;
									goto tree_2;
								}
							}
						}
					}
					else {
						if (condition_j) {
							action_4;
							goto tree_7;
						}
						else {
							if (condition_k) {
								if (condition_i) {
									if (condition_d) {
										action_5;
										goto tree_4;
									}
									else {
										action_10;
										goto tree_4;
									}
								}
								else {
									action_5;
									goto tree_4;
								}
							}
							else {
								if (condition_i) {
									action_4;
									goto tree_3;
								}
								else {
									action_2;
									goto tree_2;
								}
							}
						}
					}
				}
				else {
					if (condition_r) {
						action_6;
						goto tree_8;
					}
					else {
						action_2;
						goto tree_8;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_63: finish_condition(63);
		if (condition_o) {
			if (condition_r) {
				if (condition_j) {
					if (condition_b) {
						if (condition_i) {
							action_6;
							goto tree_73;
						}
						else {
							if (condition_c) {
								action_6;
								goto tree_73;
							}
							else {
								action_11;
								goto tree_73;
							}
						}
					}
					else {
						action_11;
						goto tree_73;
					}
				}
				else {
					if (condition_p) {
						if (condition_k) {
							if (condition_d) {
								if (condition_b) {
									if (condition_i) {
										action_6;
										goto tree_4;
									}
									else {
										if (condition_c) {
											action_6;
											goto tree_4;
										}
										else {
											action_12;
											goto tree_4;
										}
									}
								}
								else {
									action_12;
									goto tree_4;
								}
							}
							else {
								if (condition_i) {
									if (condition_b) {
										action_12;
										goto tree_4;
									}
									else {
										action_16;
										goto tree_4;
									}
								}
								else {
									action_12;
									goto tree_4;
								}
							}
						}
						else {
							if (condition_i) {
								if (condition_b) {
									action_6;
									goto tree_3;
								}
								else {
									action_11;
									goto tree_3;
								}
							}
							else {
								action_6;
								goto tree_2;
							}
						}
					}
					else {
						if (condition_i) {
							if (condition_b) {
								action_6;
								goto tree_63;
							}
							else {
								action_11;
								goto tree_63;
							}
						}
						else {
							action_6;
							goto tree_61;
						}
					}
				}
			}
			else {
				if (condition_j) {
					action_4;
					goto tree_73;
				}
				else {
					if (condition_p) {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
					else {
						if (condition_i) {
							action_4;
							goto tree_63;
						}
						else {
							action_2;
							goto tree_61;
						}
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_r) {
						if (condition_j) {
							if (condition_b) {
								if (condition_i) {
									action_6;
									goto tree_7;
								}
								else {
									if (condition_c) {
										action_6;
										goto tree_7;
									}
									else {
										action_11;
										goto tree_7;
									}
								}
							}
							else {
								action_11;
								goto tree_7;
							}
						}
						else {
							if (condition_k) {
								if (condition_d) {
									if (condition_b) {
										if (condition_i) {
											action_6;
											goto tree_4;
										}
										else {
											if (condition_c) {
												action_6;
												goto tree_4;
											}
											else {
												action_12;
												goto tree_4;
											}
										}
									}
									else {
										action_12;
										goto tree_4;
									}
								}
								else {
									if (condition_i) {
										if (condition_b) {
											action_12;
											goto tree_4;
										}
										else {
											action_16;
											goto tree_4;
										}
									}
									else {
										action_12;
										goto tree_4;
									}
								}
							}
							else {
								if (condition_i) {
									if (condition_b) {
										action_6;
										goto tree_3;
									}
									else {
										action_11;
										goto tree_3;
									}
								}
								else {
									action_6;
									goto tree_2;
								}
							}
						}
					}
					else {
						if (condition_j) {
							action_4;
							goto tree_7;
						}
						else {
							if (condition_k) {
								if (condition_i) {
									if (condition_d) {
										action_5;
										goto tree_4;
									}
									else {
										action_10;
										goto tree_4;
									}
								}
								else {
									action_5;
									goto tree_4;
								}
							}
							else {
								if (condition_i) {
									action_4;
									goto tree_3;
								}
								else {
									action_2;
									goto tree_2;
								}
							}
						}
					}
				}
				else {
					if (condition_r) {
						action_6;
						goto tree_8;
					}
					else {
						action_2;
						goto tree_8;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_73: finish_condition(73);
		if (condition_o) {
			if (condition_n) {
				if (condition_j) {
					if (condition_i) {
						action_6;
						goto tree_73;
					}
					else {
						if (condition_c) {
							action_6;
							goto tree_73;
						}
						else {
							action_11;
							goto tree_73;
						}
					}
				}
				else {
					if (condition_p) {
						if (condition_k) {
							if (condition_d) {
								if (condition_i) {
									action_6;
									goto tree_4;
								}
								else {
									if (condition_c) {
										action_6;
										goto tree_4;
									}
									else {
										action_12;
										goto tree_4;
									}
								}
							}
							else {
								action_12;
								goto tree_4;
							}
						}
						else {
							action_6;
							goto tree_47;
						}
					}
					else {
						action_6;
						goto tree_82;
					}
				}
			}
			else {
				if (condition_r) {
					if (condition_j) {
						if (condition_i) {
							action_6;
							goto tree_73;
						}
						else {
							if (condition_c) {
								action_6;
								goto tree_73;
							}
							else {
								action_11;
								goto tree_73;
							}
						}
					}
					else {
						if (condition_p) {
							if (condition_k) {
								if (condition_d) {
									if (condition_i) {
										action_6;
										goto tree_4;
									}
									else {
										if (condition_c) {
											action_6;
											goto tree_4;
										}
										else {
											action_12;
											goto tree_4;
										}
									}
								}
								else {
									action_12;
									goto tree_4;
								}
							}
							else {
								action_6;
								goto tree_47;
							}
						}
						else {
							action_6;
							goto tree_82;
						}
					}
				}
				else {
					if (condition_j) {
						if (condition_i) {
							action_4;
							goto tree_73;
						}
						else {
							if (condition_c) {
								action_4;
								goto tree_73;
							}
							else {
								action_7;
								goto tree_73;
							}
						}
					}
					else {
						if (condition_p) {
							if (condition_k) {
								if (condition_i) {
									if (condition_d) {
										action_5;
										goto tree_4;
									}
									else {
										action_10;
										goto tree_4;
									}
								}
								else {
									if (condition_d) {
										if (condition_c) {
											action_5;
											goto tree_4;
										}
										else {
											action_8;
											goto tree_4;
										}
									}
									else {
										action_8;
										goto tree_4;
									}
								}
							}
							else {
								if (condition_i) {
									action_4;
									goto tree_3;
								}
								else {
									action_3;
									goto tree_2;
								}
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_63;
							}
							else {
								action_3;
								goto tree_61;
							}
						}
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_n) {
						if (condition_j) {
							if (condition_i) {
								action_6;
								goto tree_7;
							}
							else {
								if (condition_c) {
									action_6;
									goto tree_7;
								}
								else {
									action_11;
									goto tree_7;
								}
							}
						}
						else {
							if (condition_k) {
								if (condition_d) {
									if (condition_i) {
										action_6;
										goto tree_4;
									}
									else {
										if (condition_c) {
											action_6;
											goto tree_4;
										}
										else {
											action_12;
											goto tree_4;
										}
									}
								}
								else {
									action_12;
									goto tree_4;
								}
							}
							else {
								action_6;
								goto tree_47;
							}
						}
					}
					else {
						if (condition_r) {
							if (condition_j) {
								if (condition_i) {
									action_6;
									goto tree_7;
								}
								else {
									if (condition_c) {
										action_6;
										goto tree_7;
									}
									else {
										action_11;
										goto tree_7;
									}
								}
							}
							else {
								if (condition_k) {
									if (condition_d) {
										if (condition_i) {
											action_6;
											goto tree_4;
										}
										else {
											if (condition_c) {
												action_6;
												goto tree_4;
											}
											else {
												action_12;
												goto tree_4;
											}
										}
									}
									else {
										action_12;
										goto tree_4;
									}
								}
								else {
									if (condition_i) {
										action_6;
										goto tree_3;
									}
									else {
										action_6;
										goto tree_2;
									}
								}
							}
						}
						else {
							if (condition_j) {
								action_4;
								goto tree_7;
							}
							else {
								if (condition_k) {
									if (condition_i) {
										if (condition_d) {
											action_5;
											goto tree_4;
										}
										else {
											action_10;
											goto tree_4;
										}
									}
									else {
										action_5;
										goto tree_4;
									}
								}
								else {
									if (condition_i) {
										action_4;
										goto tree_3;
									}
									else {
										action_2;
										goto tree_2;
									}
								}
							}
						}
					}
				}
				else {
					if (condition_r) {
						action_6;
						goto tree_8;
					}
					else {
						if (condition_n) {
							action_6;
							goto tree_8;
						}
						else {
							action_2;
							goto tree_8;
						}
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_82: finish_condition(82);
		if (condition_o) {
			if (condition_r) {
				if (condition_j) {
					if (condition_g) {
						if (condition_b) {
							if (condition_i) {
								action_6;
								goto tree_73;
							}
							else {
								if (condition_c) {
									action_6;
									goto tree_73;
								}
								else {
									action_11;
									goto tree_73;
								}
							}
						}
						else {
							action_11;
							goto tree_73;
						}
					}
					else {
						action_11;
						goto tree_73;
					}
				}
				else {
					if (condition_p) {
						if (condition_k) {
							if (condition_d) {
								if (condition_g) {
									if (condition_b) {
										if (condition_i) {
											action_6;
											goto tree_4;
										}
										else {
											if (condition_c) {
												action_6;
												goto tree_4;
											}
											else {
												action_12;
												goto tree_4;
											}
										}
									}
									else {
										action_12;
										goto tree_4;
									}
								}
								else {
									action_12;
									goto tree_4;
								}
							}
							else {
								if (condition_i) {
									if (condition_g) {
										if (condition_b) {
											action_12;
											goto tree_4;
										}
										else {
											action_16;
											goto tree_4;
										}
									}
									else {
										action_16;
										goto tree_4;
									}
								}
								else {
									action_12;
									goto tree_4;
								}
							}
						}
						else {
							if (condition_i) {
								if (condition_g) {
									if (condition_b) {
										action_6;
										goto tree_3;
									}
									else {
										action_11;
										goto tree_3;
									}
								}
								else {
									action_11;
									goto tree_3;
								}
							}
							else {
								action_6;
								goto tree_2;
							}
						}
					}
					else {
						if (condition_i) {
							if (condition_g) {
								if (condition_b) {
									action_6;
									goto tree_63;
								}
								else {
									action_11;
									goto tree_63;
								}
							}
							else {
								action_11;
								goto tree_63;
							}
						}
						else {
							action_6;
							goto tree_61;
						}
					}
				}
			}
			else {
				if (condition_j) {
					action_4;
					goto tree_73;
				}
				else {
					if (condition_p) {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
					else {
						if (condition_i) {
							action_4;
							goto tree_63;
						}
						else {
							action_2;
							goto tree_61;
						}
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_r) {
						if (condition_j) {
							if (condition_g) {
								if (condition_b) {
									if (condition_i) {
										action_6;
										goto tree_7;
									}
									else {
										if (condition_c) {
											action_6;
											goto tree_7;
										}
										else {
											action_11;
											goto tree_7;
										}
									}
								}
								else {
									action_11;
									goto tree_7;
								}
							}
							else {
								action_11;
								goto tree_7;
							}
						}
						else {
							if (condition_k) {
								if (condition_d) {
									if (condition_g) {
										if (condition_b) {
											if (condition_i) {
												action_6;
												goto tree_4;
											}
											else {
												if (condition_c) {
													action_6;
													goto tree_4;
												}
												else {
													action_12;
													goto tree_4;
												}
											}
										}
										else {
											action_12;
											goto tree_4;
										}
									}
									else {
										action_12;
										goto tree_4;
									}
								}
								else {
									if (condition_i) {
										if (condition_g) {
											if (condition_b) {
												action_12;
												goto tree_4;
											}
											else {
												action_16;
												goto tree_4;
											}
										}
										else {
											action_16;
											goto tree_4;
										}
									}
									else {
										action_12;
										goto tree_4;
									}
								}
							}
							else {
								if (condition_i) {
									if (condition_g) {
										if (condition_b) {
											action_6;
											goto tree_3;
										}
										else {
											action_11;
											goto tree_3;
										}
									}
									else {
										action_11;
										goto tree_3;
									}
								}
								else {
									action_6;
									goto tree_2;
								}
							}
						}
					}
					else {
						if (condition_j) {
							action_4;
							goto tree_7;
						}
						else {
							if (condition_k) {
								if (condition_i) {
									if (condition_d) {
										action_5;
										goto tree_4;
									}
									else {
										action_10;
										goto tree_4;
									}
								}
								else {
									action_5;
									goto tree_4;
								}
							}
							else {
								if (condition_i) {
									action_4;
									goto tree_3;
								}
								else {
									action_2;
									goto tree_2;
								}
							}
						}
					}
				}
				else {
					if (condition_r) {
						action_6;
						goto tree_8;
					}
					else {
						action_2;
						goto tree_8;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	break_0:
		if (condition_o) {
			if (condition_i) {
				action_4;
			}
			else {
				if (condition_h) {
					action_3;
				}
				else {
					action_2;
				}
			}
		}
		else {
			if (condition_s) {
				action_2;
			}
			else {
				action_1;
			}
		}
		continue;
	break_1:
		if (condition_o) {
			if (condition_h) {
				action_9;
			}
			else {
				if (condition_i) {
					action_11;
				}
				else {
					action_6;
				}
			}
		}
		else {
			if (condition_s) {
				action_6;
			}
			else {
				action_1;
			}
		}
		continue;
	break_2:
		if (condition_o) {
			action_6;
		}
		else {
			if (condition_s) {
				action_6;
			}
			else {
				action_1;
			}
		}
		continue;
	break_3:
		if (condition_o) {
			action_6;
		}
		else {
			if (condition_s) {
				action_6;
			}
			else {
				action_1;
			}
		}
		continue;
	break_4:
		if (condition_o) {
			action_6;
		}
		else {
			if (condition_s) {
				action_6;
			}
			else {
				action_1;
			}
		}
		continue;
	break_7:
		if (condition_o) {
			action_6;
		}
		else {
			if (condition_s) {
				action_6;
			}
			else {
				action_1;
			}
		}
		continue;
	break_8:
		if (condition_o) {
			if (condition_r) {
				if (condition_h) {
					action_9;
				}
				else {
					if (condition_i) {
						action_11;
					}
					else {
						action_6;
					}
				}
			}
			else {
				if (condition_i) {
					action_4;
				}
				else {
					if (condition_h) {
						action_3;
					}
					else {
						action_2;
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_r) {
					action_6;
				}
				else {
					action_2;
				}
			}
			else {
				action_1;
			}
		}
		continue;
	break_47:
		if (condition_o) {
			action_6;
		}
		else {
			if (condition_s) {
				action_6;
			}
			else {
				action_1;
			}
		}
		continue;
	break_61:
		if (condition_o) {
			if (condition_r) {
				if (condition_i) {
					action_11;
				}
				else {
					action_6;
				}
			}
			else {
				if (condition_i) {
					action_4;
				}
				else {
					action_2;
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_r) {
					action_6;
				}
				else {
					action_2;
				}
			}
			else {
				action_1;
			}
		}
		continue;
	break_63:
		if (condition_o) {
			if (condition_r) {
				if (condition_i) {
					if (condition_b) {
						action_6;
					}
					else {
						action_11;
					}
				}
				else {
					action_6;
				}
			}
			else {
				if (condition_i) {
					action_4;
				}
				else {
					action_2;
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_r) {
					action_6;
				}
				else {
					action_2;
				}
			}
			else {
				action_1;
			}
		}
		continue;
	break_73:
		if (condition_o) {
			if (condition_n) {
				action_6;
			}
			else {
				if (condition_r) {
					action_6;
				}
				else {
					if (condition_i) {
						action_4;
					}
					else {
						action_3;
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_r) {
					action_6;
				}
				else {
					if (condition_n) {
						action_6;
					}
					else {
						action_2;
					}
				}
			}
			else {
				action_1;
			}
		}
		continue;
	break_82:
		if (condition_o) {
			if (condition_r) {
				if (condition_i) {
					if (condition_g) {
						if (condition_b) {
							action_6;
						}
						else {
							action_11;
						}
					}
					else {
						action_11;
					}
				}
				else {
					action_6;
				}
			}
			else {
				if (condition_i) {
					action_4;
				}
				else {
					action_2;
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_r) {
					action_6;
				}
				else {
					action_2;
				}
			}
			else {
				action_1;
			}
		}
		continue;
	}

	for (int r = h - 1; r < h; r += 2) {
		// Get row pointers
		const uchar* const img_row = img.ptr<uchar>(r);
		uint* const imgLabels_row = imgLabels.ptr<uint>(r);
		uint* const imgLabels_row_prev_prev = (uint *)(((char *)imgLabels_row) - imgLabels.step.p[0] - imgLabels.step.p[0]);

		int c = -2;
		lr_finish_condition(0);
		if (condition_o) {
			if (condition_j) {
				if (condition_i) {
					action_4;
					goto lr_tree_19;
				}
				else {
					action_4;
					goto lr_tree_19;
				}
			}
			else {
				if (condition_p) {
					if (condition_k) {
						if (condition_i) {
							if (condition_d) {
								action_5;
								goto lr_tree_3;
							}
							else {
								action_10;
								goto lr_tree_3;
							}
						}
						else {
							action_5;
							goto lr_tree_3;
						}
					}
					else {
						if (condition_i) {
							action_4;
							goto lr_tree_2;
						}
						else {
							action_2;
							goto lr_tree_1;
						}
					}
				}
				else {
					if (condition_i) {
						action_4;
						goto lr_tree_7;
					}
					else {
						action_2;
						goto lr_tree_7;
					}
				}
			}
		}
		else {
			if (condition_p) {
				if (condition_j) {
					action_4;
					goto lr_tree_6;
				}
				else {
					if (condition_k) {
						if (condition_i) {
							if (condition_d) {
								action_5;
								goto lr_tree_3;
							}
							else {
								action_10;
								goto lr_tree_3;
							}
						}
						else {
							action_5;
							goto lr_tree_3;
						}
					}
					else {
						if (condition_i) {
							action_4;
							goto lr_tree_2;
						}
						else {
							action_2;
							goto lr_tree_1;
						}
					}
				}
			}
			else {
				action_1;
				goto lr_tree_0;
			}
		}
	lr_tree_0: lr_finish_condition(0);
		if (condition_o) {
			if (condition_j) {
				if (condition_i) {
					action_4;
					goto lr_tree_19;
				}
				else {
					if (condition_h) {
						if (condition_c) {
							action_4;
							goto lr_tree_19;
						}
						else {
							action_7;
							goto lr_tree_19;
						}
					}
					else {
						action_4;
						goto lr_tree_19;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_k) {
						if (condition_i) {
							if (condition_d) {
								action_5;
								goto lr_tree_3;
							}
							else {
								action_10;
								goto lr_tree_3;
							}
						}
						else {
							if (condition_h) {
								if (condition_d) {
									if (condition_c) {
										action_5;
										goto lr_tree_3;
									}
									else {
										action_8;
										goto lr_tree_3;
									}
								}
								else {
									action_8;
									goto lr_tree_3;
								}
							}
							else {
								action_5;
								goto lr_tree_3;
							}
						}
					}
					else {
						if (condition_i) {
							action_4;
							goto lr_tree_2;
						}
						else {
							if (condition_h) {
								action_3;
								goto lr_tree_1;
							}
							else {
								action_2;
								goto lr_tree_1;
							}
						}
					}
				}
				else {
					if (condition_i) {
						action_4;
						goto lr_tree_7;
					}
					else {
						if (condition_h) {
							action_3;
							goto lr_tree_7;
						}
						else {
							action_2;
							goto lr_tree_7;
						}
					}
				}
			}
		}
		else {
			if (condition_p) {
				if (condition_j) {
					action_4;
					goto lr_tree_6;
				}
				else {
					if (condition_k) {
						if (condition_i) {
							if (condition_d) {
								action_5;
								goto lr_tree_3;
							}
							else {
								action_10;
								goto lr_tree_3;
							}
						}
						else {
							action_5;
							goto lr_tree_3;
						}
					}
					else {
						if (condition_i) {
							action_4;
							goto lr_tree_2;
						}
						else {
							action_2;
							goto lr_tree_1;
						}
					}
				}
			}
			else {
				action_1;
				goto lr_tree_0;
			}
		}
	lr_tree_1: lr_finish_condition(1);
		if (condition_o) {
			if (condition_j) {
				action_11;
				goto lr_tree_19;
			}
			else {
				if (condition_p) {
					if (condition_k) {
						if (condition_d) {
							action_12;
							goto lr_tree_3;
						}
						else {
							action_12;
							goto lr_tree_3;
						}
					}
					else {
						action_6;
						goto lr_tree_24;
					}
				}
				else {
					action_6;
					goto lr_tree_7;
				}
			}
		}
		else {
			if (condition_p) {
				if (condition_j) {
					action_4;
					goto lr_tree_6;
				}
				else {
					if (condition_k) {
						action_5;
						goto lr_tree_3;
					}
					else {
						action_2;
						goto lr_tree_1;
					}
				}
			}
			else {
				action_1;
				goto lr_tree_0;
			}
		}
	lr_tree_2: lr_finish_condition(2);
		if (condition_o) {
			if (condition_j) {
				if (condition_c) {
					if (condition_b) {
						action_6;
						goto lr_tree_19;
					}
					else {
						action_11;
						goto lr_tree_19;
					}
				}
				else {
					action_11;
					goto lr_tree_19;
				}
			}
			else {
				if (condition_p) {
					if (condition_k) {
						if (condition_d) {
							if (condition_c) {
								if (condition_b) {
									action_6;
									goto lr_tree_3;
								}
								else {
									action_12;
									goto lr_tree_3;
								}
							}
							else {
								action_12;
								goto lr_tree_3;
							}
						}
						else {
							action_12;
							goto lr_tree_3;
						}
					}
					else {
						action_6;
						goto lr_tree_24;
					}
				}
				else {
					action_6;
					goto lr_tree_7;
				}
			}
		}
		else {
			if (condition_p) {
				if (condition_j) {
					action_4;
					goto lr_tree_6;
				}
				else {
					if (condition_k) {
						action_5;
						goto lr_tree_3;
					}
					else {
						action_2;
						goto lr_tree_1;
					}
				}
			}
			else {
				action_1;
				goto lr_tree_0;
			}
		}
	lr_tree_3: lr_finish_condition(3);
		if (condition_o) {
			if (condition_j) {
				action_6;
				goto lr_tree_19;
			}
			else {
				if (condition_p) {
					if (condition_k) {
						if (condition_d) {
							action_6;
							goto lr_tree_3;
						}
						else {
							action_12;
							goto lr_tree_3;
						}
					}
					else {
						action_6;
						goto lr_tree_24;
					}
				}
				else {
					action_6;
					goto lr_tree_7;
				}
			}
		}
		else {
			if (condition_p) {
				if (condition_j) {
					action_4;
					goto lr_tree_6;
				}
				else {
					if (condition_k) {
						if (condition_d) {
							action_5;
							goto lr_tree_3;
						}
						else {
							action_10;
							goto lr_tree_3;
						}
					}
					else {
						action_4;
						goto lr_tree_2;
					}
				}
			}
			else {
				action_1;
				goto lr_tree_0;
			}
		}
	lr_tree_6: lr_finish_condition(6);
		if (condition_o) {
			if (condition_j) {
				if (condition_i) {
					action_6;
					goto lr_tree_19;
				}
				else {
					if (condition_c) {
						action_6;
						goto lr_tree_19;
					}
					else {
						action_11;
						goto lr_tree_19;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_k) {
						if (condition_d) {
							if (condition_i) {
								action_6;
								goto lr_tree_3;
							}
							else {
								if (condition_c) {
									action_6;
									goto lr_tree_3;
								}
								else {
									action_12;
									goto lr_tree_3;
								}
							}
						}
						else {
							action_12;
							goto lr_tree_3;
						}
					}
					else {
						action_6;
						goto lr_tree_24;
					}
				}
				else {
					action_6;
					goto lr_tree_7;
				}
			}
		}
		else {
			if (condition_p) {
				if (condition_j) {
					action_4;
					goto lr_tree_6;
				}
				else {
					if (condition_k) {
						if (condition_i) {
							if (condition_d) {
								action_5;
								goto lr_tree_3;
							}
							else {
								action_10;
								goto lr_tree_3;
							}
						}
						else {
							action_5;
							goto lr_tree_3;
						}
					}
					else {
						if (condition_i) {
							action_4;
							goto lr_tree_2;
						}
						else {
							action_2;
							goto lr_tree_1;
						}
					}
				}
			}
			else {
				action_1;
				goto lr_tree_0;
			}
		}
	lr_tree_7: lr_finish_condition(7);
		if (condition_o) {
			if (condition_j) {
				action_4;
				goto lr_tree_19;
			}
			else {
				if (condition_p) {
					if (condition_k) {
						if (condition_i) {
							if (condition_d) {
								action_5;
								goto lr_tree_3;
							}
							else {
								action_10;
								goto lr_tree_3;
							}
						}
						else {
							action_5;
							goto lr_tree_3;
						}
					}
					else {
						if (condition_i) {
							action_4;
							goto lr_tree_2;
						}
						else {
							action_2;
							goto lr_tree_1;
						}
					}
				}
				else {
					if (condition_i) {
						action_4;
						goto lr_tree_7;
					}
					else {
						action_2;
						goto lr_tree_7;
					}
				}
			}
		}
		else {
			if (condition_p) {
				if (condition_j) {
					action_4;
					goto lr_tree_6;
				}
				else {
					if (condition_k) {
						if (condition_i) {
							if (condition_d) {
								action_5;
								goto lr_tree_3;
							}
							else {
								action_10;
								goto lr_tree_3;
							}
						}
						else {
							action_5;
							goto lr_tree_3;
						}
					}
					else {
						if (condition_i) {
							action_4;
							goto lr_tree_2;
						}
						else {
							action_2;
							goto lr_tree_1;
						}
					}
				}
			}
			else {
				action_1;
				goto lr_tree_0;
			}
		}
	lr_tree_19: lr_finish_condition(19);
		if (condition_o) {
			if (condition_n) {
				if (condition_j) {
					if (condition_i) {
						action_6;
						goto lr_tree_19;
					}
					else {
						if (condition_c) {
							action_6;
							goto lr_tree_19;
						}
						else {
							action_11;
							goto lr_tree_19;
						}
					}
				}
				else {
					if (condition_p) {
						if (condition_k) {
							if (condition_d) {
								if (condition_i) {
									action_6;
									goto lr_tree_3;
								}
								else {
									if (condition_c) {
										action_6;
										goto lr_tree_3;
									}
									else {
										action_12;
										goto lr_tree_3;
									}
								}
							}
							else {
								action_12;
								goto lr_tree_3;
							}
						}
						else {
							action_6;
							goto lr_tree_24;
						}
					}
					else {
						action_6;
						goto lr_tree_7;
					}
				}
			}
			else {
				if (condition_j) {
					if (condition_i) {
						action_4;
						goto lr_tree_19;
					}
					else {
						if (condition_c) {
							action_4;
							goto lr_tree_19;
						}
						else {
							action_7;
							goto lr_tree_19;
						}
					}
				}
				else {
					if (condition_p) {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto lr_tree_3;
								}
								else {
									action_10;
									goto lr_tree_3;
								}
							}
							else {
								if (condition_d) {
									if (condition_c) {
										action_5;
										goto lr_tree_3;
									}
									else {
										action_8;
										goto lr_tree_3;
									}
								}
								else {
									action_8;
									goto lr_tree_3;
								}
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto lr_tree_2;
							}
							else {
								action_3;
								goto lr_tree_1;
							}
						}
					}
					else {
						if (condition_i) {
							action_4;
							goto lr_tree_7;
						}
						else {
							action_3;
							goto lr_tree_7;
						}
					}
				}
			}
		}
		else {
			if (condition_p) {
				if (condition_j) {
					action_4;
					goto lr_tree_6;
				}
				else {
					if (condition_k) {
						if (condition_i) {
							if (condition_d) {
								action_5;
								goto lr_tree_3;
							}
							else {
								action_10;
								goto lr_tree_3;
							}
						}
						else {
							action_5;
							goto lr_tree_3;
						}
					}
					else {
						if (condition_i) {
							action_4;
							goto lr_tree_2;
						}
						else {
							action_2;
							goto lr_tree_1;
						}
					}
				}
			}
			else {
				action_1;
				goto lr_tree_0;
			}
		}
	lr_tree_24: lr_finish_condition(24);
		if (condition_o) {
			if (condition_j) {
				if (condition_c) {
					if (condition_g) {
						if (condition_b) {
							action_6;
							goto lr_tree_19;
						}
						else {
							action_11;
							goto lr_tree_19;
						}
					}
					else {
						action_11;
						goto lr_tree_19;
					}
				}
				else {
					action_11;
					goto lr_tree_19;
				}
			}
			else {
				if (condition_p) {
					if (condition_k) {
						if (condition_d) {
							if (condition_c) {
								if (condition_g) {
									if (condition_b) {
										action_6;
										goto lr_tree_3;
									}
									else {
										action_12;
										goto lr_tree_3;
									}
								}
								else {
									action_12;
									goto lr_tree_3;
								}
							}
							else {
								action_12;
								goto lr_tree_3;
							}
						}
						else {
							action_12;
							goto lr_tree_3;
						}
					}
					else {
						action_6;
						goto lr_tree_24;
					}
				}
				else {
					action_6;
					goto lr_tree_7;
				}
			}
		}
		else {
			if (condition_p) {
				if (condition_j) {
					action_4;
					goto lr_tree_6;
				}
				else {
					if (condition_k) {
						action_5;
						goto lr_tree_3;
					}
					else {
						action_2;
						goto lr_tree_1;
					}
				}
			}
			else {
				action_1;
				goto lr_tree_0;
			}
		}
	lr_break_0:
		if (condition_o) {
			if (condition_i) {
				action_4;
			}
			else {
				if (condition_h) {
					action_3;
				}
				else {
					action_2;
				}
			}
		}
		else {
			action_1;
		}
		continue;
	lr_break_1:
		if (condition_o) {
			action_6;
		}
		else {
			action_1;
		}
		continue;
	lr_break_2:
		if (condition_o) {
			action_6;
		}
		else {
			action_1;
		}
		continue;
	lr_break_3:
		if (condition_o) {
			action_6;
		}
		else {
			action_1;
		}
		continue;
	lr_break_6:
		if (condition_o) {
			action_6;
		}
		else {
			action_1;
		}
		continue;
	lr_break_7:
		if (condition_o) {
			if (condition_i) {
				action_4;
			}
			else {
				action_2;
			}
		}
		else {
			action_1;
		}
		continue;
	lr_break_19:
		if (condition_o) {
			if (condition_n) {
				action_6;
			}
			else {
				if (condition_i) {
					action_4;
				}
				else {
					action_3;
				}
			}
		}
		else {
			action_1;
		}
		continue;
	lr_break_24:
		if (condition_o) {
			action_6;
		}
		else {
			action_1;
		}
		continue;
	}

#undef fr_finish_condition
#undef finish_condition
#undef lr_finish_condition
}


inline static
uint secondScanSpaghetti_ee(Mat1i& imgLabels, uint* P, uint &lunique) {
	uint nLabel = flattenL(P, lunique);

	// Second scan
	for (int r = 0; r < imgLabels.rows; r += 2) {
		// Get row pointers
		uint* const imgLabels_row = imgLabels.ptr<uint>(r);
		uint* const imgLabels_row_fol = (uint *)(((char *)imgLabels_row) + imgLabels.step.p[0]);
		for (int c = 0; c<imgLabels.cols; c += 2) {
			uint iLabel = imgLabels_row[c];
			uint pix = 0xf0000000 & iLabel;
			iLabel = P[0x0fffffff & iLabel];

			imgLabels_row[c] = (0 - ((pix >> 31) & 1)) & iLabel;
			imgLabels_row[c + 1] = (0 - ((pix >> 30) & 1)) & iLabel;
			imgLabels_row_fol[c] = (0 - ((pix >> 29) & 1)) & iLabel;
			imgLabels_row_fol[c + 1] = (0 - ((pix >> 28) & 1)) & iLabel;
		}
	}

	return nLabel;
}

inline static
uint secondScanSpaghetti_oe(Mat1i& imgLabels, uint* P, uint &lunique) {
	uint nLabel = flattenL(P, lunique);

	// Second scan
	for (int r = 0; r < imgLabels.rows - 1; r += 2) {
		// Get row pointers
		uint* const imgLabels_row = imgLabels.ptr<uint>(r);
		uint* const imgLabels_row_fol = (uint *)(((char *)imgLabels_row) + imgLabels.step.p[0]);
		for (int c = 0; c<imgLabels.cols; c += 2) {
			uint iLabel = imgLabels_row[c];
			uint pix = 0xf0000000 & iLabel;
			iLabel = P[0x0fffffff & iLabel];

			imgLabels_row[c] = (0 - ((pix >> 31) & 1)) & iLabel;
			imgLabels_row[c + 1] = (0 - ((pix >> 30) & 1)) & iLabel;
			imgLabels_row_fol[c] = (0 - ((pix >> 29) & 1)) & iLabel;
			imgLabels_row_fol[c + 1] = (0 - ((pix >> 28) & 1)) & iLabel;
		}
	}
	{
		int r = imgLabels.rows - 1;
		// Get row pointers
		uint* const imgLabels_row = imgLabels.ptr<uint>(r);
		for (int c = 0; c<imgLabels.cols; c += 2) {
			uint iLabel = imgLabels_row[c];
			uint pix = 0xf0000000 & iLabel;
			iLabel = P[0x0fffffff & iLabel];

			imgLabels_row[c] = (0 - ((pix >> 31) & 1)) & iLabel;
			imgLabels_row[c + 1] = (0 - ((pix >> 30) & 1)) & iLabel;
		}
	}
	return nLabel;
}

inline static
uint secondScanSpaghetti_eo(Mat1i& imgLabels, uint* P, uint &lunique) {
	uint nLabel = flattenL(P, lunique);

	// Second scan
	for (int r = 0; r < imgLabels.rows; r += 2) {
		// Get row pointers
		uint* const imgLabels_row = imgLabels.ptr<uint>(r);
		uint* const imgLabels_row_fol = (uint *)(((char *)imgLabels_row) + imgLabels.step.p[0]);
		for (int c = 0; c < imgLabels.cols - 1; c += 2) {
			uint iLabel = imgLabels_row[c];
			uint pix = 0xf0000000 & iLabel;
			iLabel = P[0x0fffffff & iLabel];

			imgLabels_row[c] = (0 - ((pix >> 31) & 1)) & iLabel;
			imgLabels_row[c + 1] = (0 - ((pix >> 30) & 1)) & iLabel;
			imgLabels_row_fol[c] = (0 - ((pix >> 29) & 1)) & iLabel;
			imgLabels_row_fol[c + 1] = (0 - ((pix >> 28) & 1)) & iLabel;
		}
		{
			int c = imgLabels.cols - 1;
			uint iLabel = imgLabels_row[c];
			uint pix = 0xf0000000 & iLabel;
			iLabel = P[0x0fffffff & iLabel];

			imgLabels_row[c] = (0 - ((pix >> 31) & 1)) & iLabel;
			imgLabels_row_fol[c] = (0 - ((pix >> 29) & 1)) & iLabel;
		}
	}
	return nLabel;
}

inline static
uint secondScanSpaghetti_oo(Mat1i& imgLabels, uint* P, uint &lunique) {
	uint nLabel = flattenL(P, lunique);

	// Second scan
	for (int r = 0; r < imgLabels.rows - 1; r += 2) {
		// Get row pointers
		uint* const imgLabels_row = imgLabels.ptr<uint>(r);
		uint* const imgLabels_row_fol = (uint *)(((char *)imgLabels_row) + imgLabels.step.p[0]);
		for (int c = 0; c < imgLabels.cols - 1; c += 2) {
			uint iLabel = imgLabels_row[c];
			uint pix = 0xf0000000 & iLabel;
			iLabel = P[0x0fffffff & iLabel];

			imgLabels_row[c] = (0 - ((pix >> 31) & 1)) & iLabel;
			imgLabels_row[c + 1] = (0 - ((pix >> 30) & 1)) & iLabel;
			imgLabels_row_fol[c] = (0 - ((pix >> 29) & 1)) & iLabel;
			imgLabels_row_fol[c + 1] = (0 - ((pix >> 28) & 1)) & iLabel;
		}
		{
			int c = imgLabels.cols - 1;
			uint iLabel = imgLabels_row[c];
			uint pix = 0xf0000000 & iLabel;
			iLabel = P[0x0fffffff & iLabel];

			imgLabels_row[c] = (0 - ((pix >> 31) & 1)) & iLabel;
			imgLabels_row_fol[c] = (0 - ((pix >> 29) & 1)) & iLabel;
		}
	}
	{
		int r = imgLabels.rows - 1;
		// Get row pointers
		uint* const imgLabels_row = imgLabels.ptr<uint>(r);
		for (int c = 0; c < imgLabels.cols - 1; c += 2) {
			uint iLabel = imgLabels_row[c];
			uint pix = 0xf0000000 & iLabel;
			iLabel = P[0x0fffffff & iLabel];

			imgLabels_row[c] = (0 - ((pix >> 31) & 1)) & iLabel;
			imgLabels_row[c + 1] = (0 - ((pix >> 30) & 1)) & iLabel;
		}
		{
			int c = imgLabels.cols - 1;
			uint iLabel = imgLabels_row[c];
			uint pix = 0xf0000000 & iLabel;
			iLabel = P[0x0fffffff & iLabel];

			imgLabels_row[c] = (0 - ((pix >> 31) & 1)) & iLabel;
		}
	}
	return nLabel;
}

int Spaghetti(const cv::Mat1b &img, cv::Mat1i &imgLabels)
{
    imgLabels = cv::Mat1i(img.size());
	//A quick and dirty upper bound for the maximimum number of labels.
	const size_t Plength = img.rows*img.cols / 4;
	//Tree of labels
	uint *P = (uint *)fastMalloc(sizeof(uint)* Plength);
	//Background
	P[0] = 0;
	uint lunique = 1;
	uint nLabel;

	if (img.rows % 2) { // Odd rows
		if (img.cols % 2) { // Odd cols
			firstScanSpaghetti_oo(img, imgLabels, P, lunique);
			nLabel = secondScanSpaghetti_oo(imgLabels, P, lunique);
		}
		else { // Even cols
			firstScanSpaghetti_oe(img, imgLabels, P, lunique);
			nLabel = secondScanSpaghetti_oe(imgLabels, P, lunique);
		}
	}
	else { // Even rows
		if (img.cols % 2) { // Odd cols
			firstScanSpaghetti_eo(img, imgLabels, P, lunique);
			nLabel = secondScanSpaghetti_eo(imgLabels, P, lunique);
		}
		else { // Even cols
			firstScanSpaghetti_ee(img, imgLabels, P, lunique);
			nLabel = secondScanSpaghetti_ee(imgLabels, P, lunique);
		}
	}

	fastFree(P);
	return nLabel;
}
