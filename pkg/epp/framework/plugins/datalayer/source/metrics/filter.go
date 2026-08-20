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
	"bufio"
	"bytes"
	"errors"
	"io"
	"sync"
	"sync/atomic"

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

// lineReaderPool recycles the readers used to split a scrape into lines.
var lineReaderPool = sync.Pool{
	New: func() any { return bufio.NewReaderSize(nil, defaultReadBufferSize) },
}

// defaultReadBufferSize is the initial line-splitting buffer. Lines longer than
// this are handled by accumulating fragments, so the value bounds memory rather
// than the accepted line length.
const defaultReadBufferSize = 16 << 10

// errUnfilterable reports that the payload contains a line the scanner cannot
// attribute to a family, so no filtered form of it can be trusted.
var errUnfilterable = errors.New("metrics: payload contains an unrecognized line")

// filterFamilies copies the text-exposition lines belonging to want from src
// into dst.
//
// The scanner recognizes exactly one shape: a "# HELP"/"# TYPE" header or a
// sample whose name is a legacy metric name. Meeting anything else, it gives up
// on the whole payload with errUnfilterable rather than guess, and the caller
// parses the original bytes.
//
// That all-or-nothing rule is what keeps this from having to be a second
// implementation of the exposition format. The format is context-sensitive in
// places a line-oriented reader cannot see: a sample opening with '{' either
// carries its own name, quoted, or continues the family named by an earlier
// line, and telling those apart requires the grammar this scanner deliberately
// does not have. Refusing the payload costs a parse this change would otherwise
// have saved; guessing would cost a sample. The grammar it does know is the one
// that is closed by definition, so a format that grows can only ever cost the
// first.
//
// Which family a line belongs to is a separate question from which shapes are
// legible, and it is the one the filter must answer exactly: a line kept under
// the wrong family is a sample lost or a family invented. It is answered by
// familyIndex, which reproduces the parser's own attribution rather than
// approximating it.
func filterFamilies(dst *bytes.Buffer, src io.Reader, want map[string]struct{}) error {
	reader, _ := lineReaderPool.Get().(*bufio.Reader)
	reader.Reset(src)
	index, _ := indexPool.Get().(familyIndex)
	defer func() {
		reader.Reset(nil)
		lineReaderPool.Put(reader)
		clear(index)
		indexPool.Put(index)
	}()

	for {
		line, err := readLine(reader)
		if len(line) > 0 {
			keep, kerr := index.keepLine(line, want)
			if kerr != nil {
				return kerr
			}
			if keep {
				dst.Write(line)
			}
		}
		if err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}
	}
}

// filterState carries the decision made for the family a header named, so the
// samples that follow it are admitted without repeating the lookup.
// familyIndex records every family the payload has named so far and how each
// one attributes suffixed samples. It is the filter's copy of the parser's
// metricFamiliesByName, kept because which family a line belongs to is decided
// by every family declared before it, not by the one immediately above it.
type familyIndex map[string]foldKind

// indexPool recycles the per-scrape family index.
var indexPool = sync.Pool{
	New: func() any { return make(familyIndex, 64) },
}

// keepLine reports whether line survives filtering, recording what the line
// declares. It returns errUnfilterable for a line it cannot attribute.
func (idx familyIndex) keepLine(line []byte, want map[string]struct{}) (bool, error) {
	trimmed := trimLeadingSpace(line)
	if isBlank(trimmed) {
		return false, nil
	}

	var name []byte
	kind, declaresType := foldNone, false
	if trimmed[0] == '#' {
		header, headerKind, isType, ok := headerFamilyName(trimmed[1:])
		if !ok {
			// A free-form comment belongs to no family.
			return false, nil
		}
		name, kind, declaresType = header, headerKind, isType
	} else {
		name = sampleFamilyName(trimmed)
	}
	if !legacyName(name) {
		return false, errUnfilterable
	}

	family := idx.attribute(name)
	idx.record(family, kind, declaresType && string(family) == string(name))
	_, wanted := want[string(family)]
	return wanted, nil
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

// legacyName reports whether b is a metric name written in the form the text
// exposition format has always used, [a-zA-Z_:][a-zA-Z0-9_:]*.
//
// This is the one shape the scanner claims to recognize. Names the format has
// since grown other spellings for do not match, and neither will spellings it
// grows later, so they reach errUnfilterable rather than a guess.
func legacyName(b []byte) bool {
	if len(b) == 0 {
		return false
	}
	for i, c := range b {
		switch {
		case c >= 'a' && c <= 'z', c >= 'A' && c <= 'Z', c == '_', c == ':':
		case i > 0 && c >= '0' && c <= '9':
		default:
			return false
		}
	}
	return true
}

// isBlank reports whether a line carries no content.
func isBlank(line []byte) bool {
	for _, c := range line {
		if c != ' ' && c != '\t' && c != '\n' && c != '\r' {
			return false
		}
	}
	return true
}

// headerFamilyName extracts the family name from a "# HELP name ..." or
// "# TYPE name ..." line, with the leading '#' already removed. It reports
// false for any other comment. isType distinguishes the two, and kind is the
// declared type's folding behaviour, meaningful only when isType.
func headerFamilyName(comment []byte) (name []byte, kind foldKind, isType, ok bool) {
	rest := trimLeadingSpace(comment)
	keyword := rest[:fieldEnd(rest)]
	isType = bytes.Equal(keyword, []byte("TYPE"))
	if !isType && !bytes.Equal(keyword, []byte("HELP")) {
		return nil, foldNone, false, false
	}
	rest = trimLeadingSpace(rest[len(keyword):])
	name = rest[:fieldEnd(rest)]
	if len(name) == 0 {
		return nil, foldNone, false, false
	}
	if !isType {
		return name, foldNone, false, true
	}
	// The parser upper-cases the type before resolving it and accepts two
	// spellings for a gauge histogram, so the comparison has to accept every
	// spelling that reaches the folding types.
	declared := trimLeadingSpace(rest[len(name):])
	declared = declared[:fieldEnd(declared)]
	switch {
	case bytes.EqualFold(declared, []byte("summary")):
		kind = foldSummary
	case bytes.EqualFold(declared, []byte("histogram")),
		bytes.EqualFold(declared, []byte("gaugehistogram")),
		bytes.EqualFold(declared, []byte("gauge_histogram")):
		kind = foldHistogram
	}
	return name, kind, true, true
}

// sampleFamilyName returns the metric name of a sample line, which ends at the
// label list or at the whitespace before the value.
func sampleFamilyName(line []byte) []byte {
	for i := range line {
		switch line[i] {
		case '{', ' ', '\t', '\n', '\r':
			return line[:i]
		}
	}
	return line
}

func trimLeadingSpace(b []byte) []byte {
	for len(b) > 0 && (b[0] == ' ' || b[0] == '\t') {
		b = b[1:]
	}
	return b
}

// fieldEnd returns the index that ends the leading whitespace-delimited field.
func fieldEnd(b []byte) int {
	for i := range b {
		switch b[i] {
		case ' ', '\t', '\n', '\r':
			return i
		}
	}
	return len(b)
}

// readLine returns one line including its terminator. Lines longer than the
// reader's buffer are reassembled, so no scrape is rejected for line length.
// The returned slice is only valid until the next call.
func readLine(r *bufio.Reader) ([]byte, error) {
	line, err := r.ReadSlice('\n')
	if err != bufio.ErrBufferFull {
		return line, err
	}
	// Rare path: the line exceeds the buffer, so copy it out and keep reading.
	joined := make([]byte, len(line))
	copy(joined, line)
	for {
		line, err = r.ReadSlice('\n')
		joined = append(joined, line...)
		if err != bufio.ErrBufferFull {
			return joined, err
		}
	}
}
