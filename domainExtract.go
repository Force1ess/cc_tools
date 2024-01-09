package main
import (
	"bufio"
	"encoding/json"
	"io/ioutil"
	"os"
	"strings"
	"sync"
)
func errProcess(err error, message string) {
	if err != nil {
		panic(err)
	}
}

func reverseDomain(domainRev string) string {
	parts := strings.Split(domainRev, ".")
	for i, j := 0, len(parts)-1; i < j; i, j = i+1, j-1 {
		parts[i], parts[j] = parts[j], parts[i]
	}
	return strings.Join(parts, ".")
}
func domainResolve(rank_path string) {
	//"./cc-main-2022-23-sep-nov-jan-host-ranks.txt"
	fd, err := os.Open(rank_path)
	errProcess(err, "unable to open: "+rank_path)
	defer fd.Close()

	reader := bufio.NewReader(fd)

	_, err = reader.ReadString('\n')
	errProcess(err, "unable to read the line")
	var domains []string
	var wg sync.WaitGroup // modifier name type
	var lock sync.Mutex
	const domain_column = 4
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			break
		}
		fileds := strings.Fields(line)
		rev_domain := fileds[domain_column]
		wg.Add(1)
		go func() {
			defer wg.Done()
			domain := reverseDomain(rev_domain)
			lock.Lock()
			domains = append(domains, domain)
			lock.Unlock()
		}()
	}
	wg.Wait()
	saveDomains(domains, "./domains.json")
}
func saveDomains(domains []string, filename string){
	data := struct {
		Domains []string `json:"domains"`
	}{
		Domains: domains,
	}

	jsonData, err := json.MarshalIndent(data, "", "  ")
	errProcess(err, "json parse error")	

	err = ioutil.WriteFile(filename, jsonData, 0644)
	errProcess(err, "file writting error")
}
func loadDomains(filename string) ([]string) {
	data, err := ioutil.ReadFile(filename)
	errProcess(err, "unable to opem: "+filename)

	var result struct {
		Domains []string `json:"domains"`
	}
	err = json.Unmarshal(data, &result)
	errProcess(err, "unable to parse json")
	return result.Domains
}
func main(){
	file_arg := os.Args[1]
	domainResolve(file_arg)
}