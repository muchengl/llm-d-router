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
	"bytes"
	"errors"
	"io"
	"math"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/prometheus/common/model"
	"github.com/prometheus/prometheus/model/labels"
	"github.com/prometheus/prometheus/model/textparse"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

// FamilyNamer is the optional contract an extractor implements to declare the
// Prometheus metric families it reads. The metrics data source unions the
// declarations of its bound extractors and discards every other family before
// parsing. An extractor that does not implement it forces a full parse, so the
// interface is safe to adopt incrementally.
type FamilyNamer interface {
	MetricNames() []string
}

// foldKind records how a family's declared type attributes a sample whose name
// is the family name plus a series suffix. Only these two types name their
// samples that way; under any other, such a name is a family of its own.
type foldKind uint8

const (
	// foldNone is a family that claims no suffixed name: a counter, a gauge, an
	// untyped family, or one named by a sample rather than declared.
	foldNone foldKind = iota
	foldSummary
	foldHistogram
)

// The suffixes each kind claims, in the order the parser strips them.
var (
	summarySuffixes   = [...]string{"_count", "_sum"}
	histogramSuffixes = [...]string{"_count", "_sum", "_bucket"}
)

// familySelector holds the union of family names the bound extractors declared.
// Extractors bind during configuration and the set is read on every scrape, so
// the resolved set is published through an atomic pointer rather than a mutex:
// scrapes never contend with each other.
type familySelector struct {
	mu    sync.Mutex
	names map[string]struct{}
	// keepAll latches once an extractor binds without declaring anything. Such
	// an extractor may read any family, so the union of the declarations around
	// it says nothing about what the scrape is allowed to lose.
	keepAll bool
	// resolved is nil until at least one extractor declares a family. A nil
	// value means "keep everything", which is the behaviour of a data source
	// whose extractors do not implement FamilyNamer.
	resolved atomic.Pointer[map[string]struct{}]
}

// observe records the families ext declares. An extractor that declares none
// disables filtering for the source, because what it reads is unknown and
// discarding a family it needs would silently starve it.
func (s *familySelector) observe(ext fwkplugin.Plugin) {
	var names []string
	if namer, ok := ext.(FamilyNamer); ok {
		names = namer.MetricNames()
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if len(names) == 0 {
		s.keepAll = true
		s.resolved.Store(nil)
		return
	}
	if s.keepAll {
		return
	}
	if s.names == nil {
		s.names = make(map[string]struct{}, len(names))
	}
	for _, name := range names {
		if name != "" {
			s.names[name] = struct{}{}
		}
	}
	// Publish a copy so readers never observe a map being written.
	published := make(map[string]struct{}, len(s.names))
	for name := range s.names {
		published[name] = struct{}{}
	}
	s.resolved.Store(&published)
}

// wanted returns the published set, or nil when no extractor declared anything.
func (s *familySelector) wanted() map[string]struct{} {
	if p := s.resolved.Load(); p != nil {
		return *p
	}
	return nil
}

// bufferPool recycles the scratch buffers a filtered scrape needs: one holding
// the scrape as it arrived, so an unfilterable payload can still be parsed from
// the original bytes, and one holding the families an extractor reads. Both are
// recycled, so steady-state scrapes allocate nothing.
var bufferPool = sync.Pool{
	New: func() any { return new(bytes.Buffer) },
}

// errUnfilterable reports that the payload holds a construct no filtered form
// of it can be trusted to reproduce, so the caller parses the original bytes.
var errUnfilterable = errors.New("metrics: payload contains an unrecognized line")

// filterFamilies writes the entries of src belonging to want into dst, in the
// text exposition format.
//
// Reading is delegated to Prometheus's own text parser, so the grammar is not
// reimplemented here. Two things remain the filter's own work.
//
// The first is attribution. Prometheus's data model has no metric families, so
// the parser reports a sample named x_sum as a series and says nothing about
// whether it belongs to family x. Getting that wrong is not recoverable: a line
// kept under the wrong family is a sample lost or a family invented. It is
// answered by familyIndex, which reproduces expfmt's own rule. What the parser
// does supply is the declared type the rule needs, which the filter would
// otherwise have to read out of the TYPE lines itself.
//
// The second is writing. The parser reports a value as a float64, so a retained
// entry is spelled again rather than copied. The spelling can differ from the
// input where the value does not.
//
// A payload the parser rejects is refused whole with errUnfilterable rather
// than partly filtered, and the caller parses the original bytes. That keeps a
// disagreement between this parser and expfmt costing a parse rather than a
// scrape.
func filterFamilies(dst *bytes.Buffer, src []byte, want map[string]struct{}) error {
	index, _ := indexPool.Get().(familyIndex)
	defer func() {
		clear(index)
		indexPool.Put(index)
	}()

	parser := textparse.NewPromParser(src, labels.NewSymbolTable(), false)
	var lset labels.Labels
	for {
		entry, err := parser.Next()
		if err != nil {
			if errors.Is(err, io.EOF) {
				return nil
			}
			return errUnfilterable
		}

		switch entry {
		case textparse.EntryHelp:
			name, help := parser.Help()
			if !index.admit(name, foldNone, false, want) {
				continue
			}
			dst.WriteString("# HELP ")
			dst.Write(name)
			dst.WriteByte(' ')
			writeEscapedHelp(dst, help)
			dst.WriteByte('\n')

		case textparse.EntryType:
			name, declared := parser.Type()
			if !index.admit(name, foldKindOf(declared), true, want) {
				continue
			}
			dst.WriteString("# TYPE ")
			dst.Write(name)
			dst.WriteByte(' ')
			dst.WriteString(string(declared))
			dst.WriteByte('\n')

		case textparse.EntrySeries:
			series, timestamp, value := parser.Series()
			if !index.admit(seriesName(series, parser, &lset), foldNone, false, want) {
				continue
			}
			dst.Write(series)
			dst.WriteByte(' ')
			writeValue(dst, value)
			if timestamp != nil {
				dst.WriteByte(' ')
				dst.WriteString(strconv.FormatInt(*timestamp, 10))
			}
			dst.WriteByte('\n')

		case textparse.EntryComment:
			// A comment that is not a header belongs to no family, and
			// expfmt discards it.

		default:
			// A native histogram or a unit has no spelling in the format
			// expfmt reads, so no filtered form of the payload is faithful.
			return errUnfilterable
		}
	}
}

// seriesName returns the metric name of the current sample. The name heads the
// series bytes unless it is quoted inside the label list, which is the only
// spelling that needs the parser's own view of the labels.
func seriesName(series []byte, parser textparse.Parser, lset *labels.Labels) []byte {
	if len(series) > 0 && series[0] != '{' {
		for i := range series {
			switch series[i] {
			case '{', ' ', '\t':
				return series[:i]
			}
		}
		return series
	}
	parser.Labels(lset)
	return []byte(lset.Get(model.MetricNameLabel))
}

// foldKindOf maps a declared type to how it attributes a suffixed sample.
func foldKindOf(declared model.MetricType) foldKind {
	switch declared {
	case model.MetricTypeSummary:
		return foldSummary
	case model.MetricTypeHistogram, model.MetricTypeGaugeHistogram:
		return foldHistogram
	}
	return foldNone
}

// writeValue writes v as the format spells it, using the shortest form that
// parses back to v.
func writeValue(dst *bytes.Buffer, v float64) {
	switch {
	case math.IsNaN(v):
		dst.WriteString("NaN")
	case math.IsInf(v, 1):
		dst.WriteString("+Inf")
	case math.IsInf(v, -1):
		dst.WriteString("-Inf")
	default:
		var scratch [32]byte
		dst.Write(strconv.AppendFloat(scratch[:0], v, 'g', -1, 64))
	}
}

// helpEscaper restores the escapes the parser resolved when it read the text.
var helpEscaper = strings.NewReplacer(`\`, `\\`, "\n", `\n`)

func writeEscapedHelp(dst *bytes.Buffer, help []byte) {
	if !bytes.ContainsAny(help, "\\\n") {
		dst.Write(help)
		return
	}
	dst.WriteString(helpEscaper.Replace(string(help)))
}

// familyIndex records every family the payload has named so far and how each
// one attributes suffixed samples. It is the filter's copy of the parser's
// metricFamiliesByName, kept because which family a line belongs to is decided
// by every family declared before it, not by the one immediately above it.
type familyIndex map[string]foldKind

// indexPool recycles the per-scrape family index.
var indexPool = sync.Pool{
	New: func() any { return make(familyIndex, 64) },
}

// admit resolves the family name belongs to, records what the entry declares,
// and reports whether that family is wanted.
func (idx familyIndex) admit(name []byte, kind foldKind, declaresType bool, want map[string]struct{}) bool {
	family := idx.attribute(name)
	idx.record(family, kind, declaresType && string(family) == string(name))
	_, wanted := want[string(family)]
	return wanted
}

// record notes that family has been seen, and its declared type when declares
// says the type belongs to family itself. A type line resolving to some other
// family only happens when that family already has one, which the parser
// rejects outright, so there is no state to model for it.
//
// Only the entries attribute can consult are stored. It consults an entry two
// ways: by exact name, which decides a name carrying a series suffix, and as
// the base of a suffix, which only ever matters for a summary or histogram.
// Because foldNone is the zero value, an absent entry answers the second the
// same way a stored one would, so every other family can be left out. That
// keeps a scrape's cost to the few families that can take part in folding
// rather than one map insertion per family.
func (idx familyIndex) record(family []byte, kind foldKind, declares bool) {
	if declares && kind != foldNone {
		idx[string(family)] = kind
		return
	}
	if _, ok := cutSeriesSuffix(family, histogramSuffixes[:]); !ok {
		return
	}
	if _, seen := idx[string(family)]; !seen {
		idx[string(family)] = foldNone
	}
}

// attribute resolves the family a line naming name belongs to, by the rule
// TextParser.setOrCreateCurrentMF applies: a name already standing as a family
// is its own; otherwise a summary or histogram family whose name it extends
// with a series suffix claims it; otherwise it starts a family of its own.
//
// Reproducing the rule rather than approximating it is the point. An
// approximation that folds one suffix too many discards a family the parser
// would have produced, and one that folds one too few invents a family it would
// not have.
func (idx familyIndex) attribute(name []byte) []byte {
	if _, seen := idx[string(name)]; seen {
		return name
	}
	if base, ok := cutSeriesSuffix(name, summarySuffixes[:]); ok && idx[string(base)] == foldSummary {
		return base
	}
	if base, ok := cutSeriesSuffix(name, histogramSuffixes[:]); ok && idx[string(base)] == foldHistogram {
		return base
	}
	return name
}

// cutSeriesSuffix strips the first of suffixes that name carries, leaving a
// non-empty base.
func cutSeriesSuffix(name []byte, suffixes []string) ([]byte, bool) {
	for _, suffix := range suffixes {
		if len(name) > len(suffix) && string(name[len(name)-len(suffix):]) == suffix {
			return name[:len(name)-len(suffix)], true
		}
	}
	return nil, false
}
