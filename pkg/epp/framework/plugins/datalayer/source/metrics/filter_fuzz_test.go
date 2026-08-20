/*
Copyright 2026 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package metrics

import (
	"io"
	"strings"
	"testing"

	"google.golang.org/protobuf/proto"
)

// fuzzWantedNames are the families the differential target asks for. They are
// short and appear in the seeds so mutation keeps producing payloads that
// exercise the matching paths rather than only the reject paths.
//
// The suffixed names are there to catch attribution errors in both directions.
// Whether a suffixed name is a family of its own or a sample of a family named
// earlier is decided by that family's type, so asking for the base name catches
// a filter that invents a family the parser folds away, and asking for the
// suffixed name catches one that discards a family the parser would produce.
var fuzzWantedNames = []string{"a", "h", "a_sum", "h_bucket", "vllm:num_requests_waiting"}

// parseRecovered runs parse over payload, turning a panic into a value. The
// parser does not survive every byte string a scrape target can return (a
// payload of "{}" is enough), and a payload it cannot survive is outside the
// property this target checks. Keeping the two apart is what lets the target
// say whether a crash came from the payload or from filtering it.
func parseRecovered(parse func(io.Reader) (PrometheusMetricMap, error), payload string) (
	fams PrometheusMetricMap, err error, fatal any) {
	defer func() { fatal = recover() }()
	fams, err = parse(strings.NewReader(payload))
	return fams, err, nil
}

// FuzzFilterAgreesWithFullParse pins the only property the filter owes its
// caller: for the families an extractor asked for, parsing the filtered payload
// yields what parsing the whole payload would have yielded.
//
// The oracle is the parser itself, so the property is checked against the
// definition of the format rather than against a second reading of it. This is
// what keeps the line scanner honest as the exposition format grows: a shape it
// does not understand shows up here as a disagreement with expfmt.
func FuzzFilterAgreesWithFullParse(f *testing.F) {
	seeds := []string{
		"",
		"# HELP a help.\n# TYPE a gauge\na 1\n",
		"# HELP a help.\n# TYPE a gauge\na 1\n# HELP b help.\n# TYPE b gauge\nb 2\n",
		"# TYPE h histogram\nh_bucket{le=\"1\"} 1\nh_sum 2\nh_count 3\n",
		"a{note=\"{b}\"} 1\nb 2\n",
		"a 1\na_extra 2\n",
		// A series suffix is a family of its own under every type but histogram
		// and summary, which name their own samples that way.
		"a 1\na_sum 2\n",
		"# TYPE a gauge\na 1\na_sum 2\n",
		"# TYPE h histogram\nh_bucket{le=\"1\"} 1\nh_sum 2\n",
		"# TYPE a summary\na{quantile=\"0.5\"} 1\na_sum 2\na_count 3\n",
		// A summary claims _count and _sum but not _bucket.
		"# TYPE a summary\na{quantile=\"0.5\"} 1\na_sum 2\na_bucket 3\n",
		// gaugehistogram claims the same suffixes histogram does.
		"# TYPE h gaugehistogram\nh_bucket{le=\"1\"} 1\nh_sum 2\n",
		"# TYPE h gauge_histogram\nh_bucket{le=\"1\"} 1\nh_sum 2\n",
		// Attribution looks at every family named so far, not the last one.
		"# TYPE h histogram\nh_bucket{le=\"1\"} 1\n# TYPE a gauge\na 5\nh_sum 2\n",
		// A family already named shadows the folding of its own name.
		"h_bucket 1\n# TYPE h histogram\nh_bucket{le=\"1\"} 2\nh_sum 3\n",
		"a_sum 1\n# TYPE a summary\na{quantile=\"0.5\"} 2\na_sum 3\n",
		"# a free-form comment\n\na 1\n",
		"a{l=\"\\\"quoted\\\"\"} 1\n",
		"a{l=\"line\\nbreak\"} 1\n",
		"vllm:num_requests_waiting 5\n",
		// The quoted-name form the exposition format grew for names that are
		// not expressible as legacy names.
		"{\"vllm:num_requests_waiting\",le=\"1\"} 5\n",
		// A sample that continues the family an earlier line named, which
		// reads as a family of its own once that line is gone.
		"a{x=\"y\"} 1\n{x=\"z\"} 2\n",
		"A{}0\n{}0\n",
		"# HELP a help.\n# TYPE a gauge\na 1 1395066363000\n",
		"a 1 # {trace_id=\"abc\"} 1.0\n",
		"a NaN\na_extra +Inf\n",
		// A payload the parser panics on rather than rejects.
		"{}",
	}
	for _, seed := range seeds {
		f.Add(seed)
	}

	f.Fuzz(func(t *testing.T, payload string) {
		full, err, fatal := parseRecovered(parseMetrics, payload)
		if err != nil || fatal != nil {
			// The filter owes nothing on a payload the parser rejects, and
			// nothing on one it cannot survive.
			t.Skip()
		}

		parser := newFamilyFilteringParser()
		parser.observeExtractor(namerStub{names: fuzzWantedNames})
		filtered, err, fatal := parseRecovered(parser.parse, payload)
		if fatal != nil {
			t.Fatalf("filtering produced a payload the parser cannot survive: %v\npayload: %q", fatal, payload)
		}
		if err != nil {
			t.Fatalf("filtering turned an acceptable payload into a rejected one: %v\npayload: %q", err, payload)
		}

		for _, name := range fuzzWantedNames {
			want, inFull := full[name]
			got, inFiltered := filtered[name]
			switch {
			case inFull && !inFiltered:
				t.Fatalf("family %q was dropped by filtering\npayload: %q", name, payload)
			case !inFull && inFiltered:
				t.Fatalf("family %q was invented by filtering\npayload: %q", name, payload)
			case inFull && !proto.Equal(want, got):
				t.Fatalf("family %q changed under filtering\npayload: %q\n full: %v\n got: %v",
					name, payload, want, got)
			}
		}
	})
}
