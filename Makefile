CC        ?= clang
CFLAGS    ?= -fPIC -std=c17 -D_POSIX_C_SOURCE=199309L
LIB_PATHS ?=
LDFLAGS   ?= -lopenblas -lm

WARNFLAGS ?= -Wall -Wextra
CFLAGS    += $(WARNFLAGS)

SRC_DIR    = $(CURDIR)/src
BUILD_DIR  = $(CURDIR)/build
OBJ_DIR    = $(BUILD_DIR)/obj
SRCS = $(shell find $(SRC_DIR) -name '*.c')
OBJS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRCS))
BINARY     = gemm 

MKDIR ?= @mkdir -p

ifdef FAST 
	CFLAGS += -O3 -march=native
endif

ifdef OPENBLAS_PATH
	CFLAGS += -I$(OPENBLAS_PATH)/include
	LIB_PATHS += -L$(OPENBLAS_PATH)/lib
endif

.PHONY: default build clean

default: build

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	$(MKDIR) $(dir $(OBJS))
	@$(CC) $(CFLAGS) -c $< -o $@

build: $(OBJS)
	$(MKDIR) $(BUILD_DIR)
	@$(CC) $(CFLAGS) $(OBJS) $(LIB_PATHS) -o $(BUILD_DIR)/$(BINARY) $(LDFLAGS)
	@$(CC) $(CFLAGS) $(OBJS) $(LIB_PATHS) -shared -o $(BUILD_DIR)/libgemm.so $(LDFLAGS)

clean: 
	rm -rf $(BUILD_DIR)
