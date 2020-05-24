/**
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// These are helper functions for the SDK samples (string parsing, timers, image helpers, etc)
#ifndef HELPER_FUNCTIONS_H
#define HELPER_FUNCTIONS_H

#ifdef WIN32
#pragma warning(disable:4996)
#endif

// includes, project
#include "../../../../../../../../../../../usr/include/stdio.h"
#include "../../../../../../../../../../../usr/include/c++/7/stdlib.h"
#include "../../../../../../../../../../../usr/include/c++/7/string"
#include "../../../../../../../../../../../usr/include/assert.h"
#include "exception.h"
#include "../../../../../../../../../../../usr/include/c++/7/math.h"

#include "../../../../../../../../../../../usr/include/c++/7/fstream"
#include "../../../../../../../../../../../usr/include/c++/7/vector"
#include "../../../../../../../../../../../usr/include/c++/7/iostream"
#include "../../../../../../../../../../../usr/include/c++/7/algorithm"

// includes, timer, string parsing, image helpers
#include "helper_timer.h"   // helper functions for timers
#include "helper_string.h"  // helper functions for string parsing
#include "helper_image.h"   // helper functions for image compare, dump, data comparisons

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

#endif //  HELPER_FUNCTIONS_H
