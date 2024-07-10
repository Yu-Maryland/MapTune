#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <regex.h>

#define CHUNK_SIZE 1024

typedef struct {
    int num_arms;
    double epsilon;
    double *q_values;
    int *counts;
    int sample_gate;
} EpsilonGreedyMAB;

typedef struct {
    double delay;
    double area;
} Result;

EpsilonGreedyMAB* initialize_epsilon_greedy_mab(int num_arms, double epsilon, int sample_gate);
void select_action(EpsilonGreedyMAB *mab, int *selected_cells);
void update(EpsilonGreedyMAB *mab, int *selected_arms, int selected_count, double reward);
void free_epsilon_greedy_mab(EpsilonGreedyMAB *mab);
Result technology_mapper(const char *genlib_origin, int *partial_cell_library, int partial_count, const char *design, const char *lib_path, const char *temp_blif, const char *lib_origin);
double calculate_reward(double max_delay, double max_area, double delay, double area);

int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <sample_gate> <design> <genlib_origin>\n", argv[0]);
        return 1;
    }

    char *genlib_origin = argv[argc - 1];
    char *design = argv[argc - 2];
    int sample_gate = atoi(argv[argc - 3]);

    size_t genlib_origin_len = strlen(genlib_origin);
    char lib_origin[genlib_origin_len - 6];
    strncpy(lib_origin, genlib_origin, genlib_origin_len - 7);
    lib_origin[genlib_origin_len - 7] = '\0';
    strcat(lib_origin, ".lib");

    size_t design_len = strlen(design);
    char temp_blif[1024];
    snprintf(temp_blif, sizeof(temp_blif), "/export/mliu9867/LibGame/temp_blifs/%.*s_temp.blif", (int)(design_len - 6), design);

    char lib_path[] = "/export/mliu9867/LibGame/gen_newlibs/";

    char abc_cmd[2048];
    snprintf(abc_cmd, sizeof(abc_cmd),
             "read %s; read %s; map; write %s; read %s; read -m %s; ps; topo; upsize; dnsize; stime;",
             genlib_origin, design, temp_blif, lib_origin, temp_blif);

    char cmd[2080];
    snprintf(cmd, sizeof(cmd), "abc -c \"%s\"", abc_cmd);

    // Print the combined command for debugging
    printf("Combined command: %s\n", cmd);

    FILE *fp;
    fp = popen(cmd, "r");
    if (fp == NULL) {
        perror("popen failed");
        return 1;
    }

    size_t buffer_size = CHUNK_SIZE;
    char *res = malloc(buffer_size);
    if (!res) {
        perror("malloc failed");
        pclose(fp);
        return 1;
    }

    size_t total_len = 0;
    char buffer[CHUNK_SIZE];
    while (fgets(buffer, CHUNK_SIZE, fp) != NULL) {
        size_t buffer_len = strlen(buffer);
        if (total_len + buffer_len + 1 >= buffer_size) {
            buffer_size *= 2;
            res = realloc(res, buffer_size);
            if (!res) {
                perror("realloc failed");
                pclose(fp);
                return 1;
            }
        }
        strcpy(res + total_len, buffer);
        total_len += buffer_len;
    }
    pclose(fp);

    // Null-terminate the captured output
    res[total_len] = '\0';

    // Print the captured output for debugging
    printf("Captured output:\n%s\n", res);

    regex_t regex_d, regex_a;
    regmatch_t match[2];
    double max_delay = 0.0, max_area = 0.0;

    // Compile regular expressions
    if (regcomp(&regex_d, "Delay\\s*=\\s*([0-9]+\\.[0-9]+)\\s*ps", REG_EXTENDED) ||
        regcomp(&regex_a, "Area\\s*=\\s*([0-9]+\\.[0-9]+)", REG_EXTENDED)) {
        fprintf(stderr, "Could not compile regex\n");
        free(res);
        return 1;
    }

    // Execute regular expressions
    if (!regexec(&regex_d, res, 2, match, 0)) {
        char delay_str[16];
        snprintf(delay_str, sizeof(delay_str), "%.*s",
                 (int)(match[1].rm_eo - match[1].rm_so), res + match[1].rm_so);
        max_delay = atof(delay_str);
    } else {
        fprintf(stderr, "No match for delay\n");
    }

    if (!regexec(&regex_a, res, 2, match, 0)) {
        char area_str[16];
        snprintf(area_str, sizeof(area_str), "%.*s",
                 (int)(match[1].rm_eo - match[1].rm_so), res + match[1].rm_so);
        max_area = atof(area_str);
    } else {
        fprintf(stderr, "No match for area\n");
    }

    // Free compiled regular expressions
    regfree(&regex_d);
    regfree(&regex_a);
    free(res);

    printf("Baseline Delay: %f\n", max_delay);
    printf("Baseline Area: %f\n", max_area);

    // Additional code for EpsilonGreedyMAB

    // Read the file and count the number of valid lines
    fp = fopen(genlib_origin, "r");
    if (fp == NULL) {
        perror("Error opening file");
        return 1;
    }

    char **f_lines = malloc(sizeof(char *) * 1024);
    int f_lines_count = 0;
    char line[1024];
    while (fgets(line, sizeof(line), fp)) {
        if (strstr(line, "GATE") && !strstr(line, "BUF") && !strstr(line, "INV") && !strstr(line, "inv") && !strstr(line, "buf")) {
            f_lines[f_lines_count] = strdup(line);
            f_lines_count++;
        }
    }
    fclose(fp);

    int num_arms = f_lines_count;
    EpsilonGreedyMAB *mab = initialize_epsilon_greedy_mab(num_arms, 0.2, sample_gate);

    int *best_cells = NULL;
    Result best_result = {DBL_MAX, DBL_MAX};
    double best_reward = -DBL_MAX;  // Track best reward

    // Main Loop
    int num_iterations = 100;
    for (int i = 0; i < num_iterations; i++) {
        printf("Iteration: %d\n", i);
        int selected_cells[sample_gate];
        select_action(mab, selected_cells);
        Result result = technology_mapper(genlib_origin, selected_cells, sample_gate, design, lib_path, temp_blif, lib_origin);

        double reward;
        if (isnan(result.delay) || isnan(result.area)) {
            reward = -DBL_MAX;
        } else {
            reward = calculate_reward(max_delay, max_area, result.delay, result.area);
        }

        if (reward > best_reward) {
            best_reward = reward;
            printf("Current best reward: %f\n", best_reward);
            best_result = result;
            printf("Current best result: Delay = %f, Area = %f\n", best_result.delay, best_result.area);
            best_cells = malloc(sample_gate * sizeof(int));
            memcpy(best_cells, selected_cells, sample_gate * sizeof(int));
        }

        update(mab, selected_cells, sample_gate, reward);
    }

    printf("Best Cells: ");
    for (int i = 0; i < sample_gate; i++) {
        printf("%d ", best_cells[i]);
    }
    printf("\nBest Delay: %f\nBest Area: %f\nBest Reward: %f\n", best_result.delay, best_result.area, best_reward);

    free(best_cells);
    for (int i = 0; i < f_lines_count; i++) {
        free(f_lines[i]);
    }
    free(f_lines);
    free_epsilon_greedy_mab(mab);

    return 0;
}

EpsilonGreedyMAB* initialize_epsilon_greedy_mab(int num_arms, double epsilon, int sample_gate) {
    EpsilonGreedyMAB *mab = (EpsilonGreedyMAB *)malloc(sizeof(EpsilonGreedyMAB));
    mab->num_arms = num_arms;
    mab->epsilon = epsilon;
    mab->q_values = (double *)malloc(num_arms * sizeof(double));
    mab->counts = (int *)malloc(num_arms * sizeof(int));
    mab->sample_gate = sample_gate;

    for (int i = 0; i < num_arms; i++) {
        mab->q_values[i] = 0.0;
        mab->counts[i] = 0;
    }

    srand(time(NULL)); // Seed the random number generator

    return mab;
}

void select_action(EpsilonGreedyMAB *mab, int *selected_cells) {
    int selected_count = 0;
    while (selected_count < mab->sample_gate) {
        int select;
        double rand_val = (double)rand() / RAND_MAX;
        if (rand_val > mab->epsilon) {
            // Exploitation
            int max_index = 0;
            for (int i = 1; i < mab->num_arms; i++) {
                if (mab->q_values[i] > mab->q_values[max_index]) {
                    max_index = i;
                }
            }
            select = max_index;
        } else {
            // Exploration
            select = rand() % mab->num_arms;
        }

        // Check if the selected arm is already in selected_cells
        int already_selected = 0;
        for (int i = 0; i < selected_count; i++) {
            if (selected_cells[i] == select) {
                already_selected = 1;
                break;
            }
        }

        if (!already_selected) {
            selected_cells[selected_count++] = select;
        }
    }
}

void update(EpsilonGreedyMAB *mab, int *selected_arms, int selected_count, double reward) {
    for (int i = 0; i < selected_count; i++) {
        int arm = selected_arms[i];
        mab->counts[arm] += 1;
        mab->q_values[arm] = (mab->q_values[arm] * (mab->counts[arm] - 1) + reward) / mab->counts[arm];
    }
}

void free_epsilon_greedy_mab(EpsilonGreedyMAB *mab) {
    free(mab->q_values);
    free(mab->counts);
    free(mab);
}

Result technology_mapper(const char *genlib_origin, int *partial_cell_library, int partial_count, const char *design, const char *lib_path, const char *temp_blif, const char *lib_origin) {
    // Read the genlib file and filter lines
    FILE *fp = fopen(genlib_origin, "r");
    if (fp == NULL) {
        perror("Error opening file");
        Result result = {NAN, NAN};
        return result;
    }

    char **f_lines = malloc(sizeof(char *) * 1024);
    char **f_keep = malloc(sizeof(char *) * 1024);
    int f_lines_count = 0;
    int f_keep_count = 0;

    char line[1024];
    while (fgets(line, sizeof(line), fp)) {
        line[strcspn(line, "\n")] = 0; // Remove newline character
        if (strstr(line, "GATE") && !strstr(line, "BUF") && !strstr(line, "INV") && !strstr(line, "inv") && !strstr(line, "buf")) {
            f_lines[f_lines_count] = strdup(line);
            f_lines_count++;
        } else if (strstr(line, "BUF") || strstr(line, "INV") || strstr(line, "inv") || strstr(line, "buf")) {
            f_keep[f_keep_count] = strdup(line);
            f_keep_count++;
        }
    }
    fclose(fp);

    // Build the new library with partial cells
    char **lines_partial = malloc(sizeof(char *) * (partial_count + f_keep_count));
    for (int i = 0; i < partial_count; i++) {
        lines_partial[i] = strdup(f_lines[partial_cell_library[i]]);
    }
    for (int i = 0; i < f_keep_count; i++) {
        lines_partial[partial_count + i] = strdup(f_keep[i]);
    }

    char output_genlib_file[1024];
    snprintf(output_genlib_file, sizeof(output_genlib_file), "%s%s_%d_samplelib.genlib", lib_path, design, partial_count + f_keep_count);
    fp = fopen(output_genlib_file, "w");
    if (fp == NULL) {
        perror("Error opening output file");
        Result result = {NAN, NAN};
        return result;
    }

    for (int i = 0; i < partial_count + f_keep_count; i++) {
        fprintf(fp, "%s\n", lines_partial[i]);
        free(lines_partial[i]);
    }
    fclose(fp);

    free(f_lines);
    free(f_keep);
    free(lines_partial);

    // Run the command with the new library
    char abc_cmd[2048];
    snprintf(abc_cmd, sizeof(abc_cmd),
             "read %s;read %s; map; write %s; read %s;read -m %s; ps; topo; upsize; dnsize; stime;",
             output_genlib_file, design, temp_blif, lib_origin, temp_blif);

    char cmd[2080];
    snprintf(cmd, sizeof(cmd), "abc -c \"%s\"", abc_cmd);

    FILE *fp_cmd = popen(cmd, "r");
    if (fp_cmd == NULL) {
        perror("popen failed");
        Result result = {NAN, NAN};
        return result;
    }

    size_t buffer_size = CHUNK_SIZE;
    char *res = malloc(buffer_size);
    if (!res) {
        perror("malloc failed");
        pclose(fp_cmd);
        Result result = {NAN, NAN};
        return result;
    }

    size_t total_len = 0;
    char buffer[CHUNK_SIZE];
    while (fgets(buffer, CHUNK_SIZE, fp_cmd) != NULL) {
        size_t buffer_len = strlen(buffer);
        if (total_len + buffer_len + 1 >= buffer_size) {
            buffer_size *= 2;
            res = realloc(res, buffer_size);
            if (!res) {
                perror("realloc failed");
                pclose(fp_cmd);
                Result result = {NAN, NAN};
                return result;
            }
        }
        strcpy(res + total_len, buffer);
        total_len += buffer_len;
    }
    pclose(fp_cmd);

    res[total_len] = '\0';

    regex_t regex_d, regex_a;
    regmatch_t match[2];
    double delay = NAN, area = NAN;

    if (regcomp(&regex_d, "Delay\\s*=\\s*([0-9]+\\.[0-9]+)\\s*ps", REG_EXTENDED) ||
        regcomp(&regex_a, "Area\\s*=\\s*([0-9]+\\.[0-9]+)", REG_EXTENDED)) {
        fprintf(stderr, "Could not compile regex\n");
        free(res);
        Result result = {NAN, NAN};
        return result;
    }

    if (!regexec(&regex_d, res, 2, match, 0)) {
        char delay_str[16];
        snprintf(delay_str, sizeof(delay_str), "%.*s",
                 (int)(match[1].rm_eo - match[1].rm_so), res + match[1].rm_so);
        delay = atof(delay_str);
    } else {
        fprintf(stderr, "No match for delay\n");
    }

    if (!regexec(&regex_a, res, 2, match, 0)) {
        char area_str[16];
        snprintf(area_str, sizeof(area_str), "%.*s",
                 (int)(match[1].rm_eo - match[1].rm_so), res + match[1].rm_so);
        area = atof(area_str);
    } else {
        fprintf(stderr, "No match for area\n");
    }

    regfree(&regex_d);
    regfree(&regex_a);
    free(res);

    Result result = {delay, area};
    return result;
}

double calculate_reward(double max_delay, double max_area, double delay, double area) {
    double normalized_delay = delay / max_delay;
    double normalized_area = area / max_area;

    return -sqrt(normalized_delay * normalized_area);
}

