CC        ?= clang
CFLAGS    ?= -O2 -fPIC -std=c11
LIB_PATHS ?=
LDFLAGS   ?= -lopenblas

WARNFLAGS ?= -Wall -Wextra
CFLAGS    += $(WARNFLAGS)

SRC_DIR    = $(CURDIR)/src
BUILD_DIR  = $(CURDIR)/build
OBJ_DIR    = $(BUILD_DIR)/obj
SRCS = $(shell find $(SRC_DIR) -name '*.c')
OBJS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRCS))
BINARY     = runner 

MKDIR ?= @mkdir -p

ifdef FAST 
	CFLAGS += -ffast-math -march=native -funroll-loops
endif

ifdef OPENBLAS_PATH
	CFLAGS += -I$(OPENBLAS_PATH)/include
	LIB_PATHS += -L$(OPENBLAS_PATH)/lib
endif

.PHONY: default build clean run

default: build

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	$(MKDIR) $(dir $(OBJS))
	@$(CC) $(CFLAGS) -c $< -o $@

build: $(OBJS)
	$(MKDIR) $(BUILD_DIR)
	@$(CC) $(CFLAGS) $(OBJS) $(LIB_PATHS) -o $(BUILD_DIR)/$(BINARY) $(LDFLAGS)

run: build
	@OMP_NUM_THREADS=1 $(BUILD_DIR)/$(BINARY)

clean: 
	rm -rf $(BUILD_DIR)
