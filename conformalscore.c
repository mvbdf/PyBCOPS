// use: gcc -fpic -shared -o libscore.so conformalscore.c

void conformalscore(float *ste, float *s, int* count, int size_ste, int size_s) {
    for(int i = 0; i < size_ste; i++)
        for(int j = 0; j < size_s; j++)
            if(ste[i] >= s[j])
                count[i]++;
}
