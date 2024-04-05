using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using Benchmarks;
using MultiTargetLib;

namespace TokenizerBenchmarks;

[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net481)]
[SimpleJob(RuntimeMoniker.Net80)]
[SimpleJob(RuntimeMoniker.Net60)]
public class Pretokenization
{
    private readonly string _corpusPath = "data/wiki-simple.json";

    private string[] _corpus = null!;

    [GlobalSetup]
    public async Task SetupAsync()
    {
        _corpus = await CorpusReader.ReadJsonCorpusAsync(_corpusPath);
    }

    [Benchmark(Baseline = true)]
    public void RefStructEnumerator()
    {
        foreach (var input in _corpus)
        {
            foreach (var pivot in new PreTokenizingEnumerator(input, false, System.Text.NormalizationForm.FormD))
            {
                var x = pivot.Segment;
            }
        }
    }

    [Benchmark]
    public void RefStructEnumeratorNoAggrInlining()
    {
        foreach (var input in _corpus)
        {
            foreach (var pivot in new PreTokenizingEnumeratorNoAggrInlining(input, false, System.Text.NormalizationForm.FormD))
            {
                var x = pivot.Segment;
            }
        }
    }

    [Benchmark]
    public void RegexPublicMlNetNuget()
    {
        var ws = new Microsoft.ML.Tokenizers.WhiteSpace();
        foreach (var input in _corpus)
        {
            foreach (var pivot in ws.PreTokenize(input))
            {
                var x = pivot.TokenString;
            }
        }
    }

    [Benchmark]
    public void RegexCurrentMlNetGithub()
    {
        var ws = new MultiTargetLib.MLTokenizers.WhiteSpace();
        foreach (var input in _corpus)
        {
            foreach (var pivot in ws.PreTokenize(input))
            {
                var x = pivot.TokenSpan;
            }
        }
    }
}
